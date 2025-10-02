# stdlib
import time
import re
import logging
import socket
import datetime as _dt
from typing import Optional, Tuple, Dict, Any, List

# third-party
import numpy as np
import pandas as pd
from passlib.context import CryptContext
from sqlalchemy import (
    create_engine,
    text,
    Table, Column, Integer, BigInteger, String, Boolean, TIMESTAMP,
    MetaData,
    select, insert, delete, update,
)

# Streamlit secrets for config
import streamlit as st

# ---- password hashing context (Passlib handles hashing & verify) ----
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],   # single well-supported scheme
    deprecated="auto",
)

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
#                     CONFIG FROM SECRETS (STRICT)
# ============================================================
def require_secret(path: List[str], hint: str = "") -> Any:
    """
    Fetch nested keys from st.secrets; raise RuntimeError if missing.
    Example paths: ["mysql","host"], ["mysql","timeouts","connect"]
    """
    try:
        ref = st.secrets
        for k in path:
            ref = ref[k]
        return ref
    except Exception as e:
        pretty = "']['".join(path)
        raise RuntimeError(
            f"Missing st.secrets['{pretty}']. "
            f"Add it to .streamlit/secrets.toml. {hint}"
        ) from e

DB_HOST   = require_secret(["mysql", "host"],   "Example: host = \"db.example.com\"")
DB_PORT   = int(require_secret(["mysql", "port"], "Example: port = 3306"))
DB_USER   = require_secret(["mysql", "user"])
DB_PASS   = require_secret(["mysql", "password"])
DB_NAME   = require_secret(["mysql", "database"])
DB_DRIVER = st.secrets["mysql"].get("driver", "mysql+pymysql")

DB_TABLE           = st.secrets["mysql"].get("table_sales", "sales_report")
DB_TABLE_BOOKINGS  = st.secrets["mysql"].get("table_bookings", "booking_report")

# Timeouts (seconds)
CONNECT_TIMEOUT = int(st.secrets["mysql"].get("timeouts", {}).get("connect", 8))
READ_TIMEOUT    = int(st.secrets["mysql"].get("timeouts", {}).get("read", 90))
WRITE_TIMEOUT   = int(st.secrets["mysql"].get("timeouts", {}).get("write", 30))
POOL_TIMEOUT    = int(st.secrets["mysql"].get("timeouts", {}).get("pool", 10))

# Optional SSL dict: {"ca": "...", "cert": "...", "key": "..."}
SSL_DICT = st.secrets["mysql"].get("ssl", None)

# Clip to current month start
CLIP_TO_MONTH = pd.Timestamp(_dt.date.today().replace(day=1))

# Accept 2023-Jan, 2024-Nov, … ; skip YTD/TOTAL/Projected
MONTH_HEADER_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<mon>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$",
    re.IGNORECASE,
)
EXCLUDE_TOKENS = ("YTD", "TOTAL", "PROJECTED")


def _preflight_mysql(host: str, port: int, timeout: int = 3) -> None:
    """Fail fast if host:port is unreachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as e:
        raise RuntimeError(
            f"Cannot reach MySQL at {host}:{port} ({e}). "
            "Check network, firewall/SecGroup, and that host/port in st.secrets are correct."
        )


class Conn:
    """
    Usage:
        conn = Conn()
        sales = conn.load_sales_data()
        bookings = conn.load_bookings_data()

        # Auth
        conn.create_user("CAROL", "password123", role="salesperson")
        ok, role = conn.authenticate_user("CAROL", "password123")

        # Audit
        conn.write_audit("CAROL", "login", extra="Login success")
    """
    def __init__(self):
        # Preflight reachability
        _preflight_mysql(DB_HOST, DB_PORT)

        # Build SQLAlchemy URL (include port)
        host_port = f"{DB_HOST}:{DB_PORT}" if DB_PORT else DB_HOST
        url = f"{DB_DRIVER}://{DB_USER}:{DB_PASS}@{host_port}/{DB_NAME}"

        # connect_args for PyMySQL
        connect_args = {
            "connect_timeout": CONNECT_TIMEOUT,
            "read_timeout": READ_TIMEOUT,
            "write_timeout": WRITE_TIMEOUT,
        }
        if SSL_DICT:
            # PyMySQL expects a dict under key 'ssl'
            connect_args["ssl"] = dict(SSL_DICT)

        t0 = time.perf_counter()
        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=3,
            max_overflow=2,
            pool_timeout=POOL_TIMEOUT,
            connect_args=connect_args,
        )
        self.table = DB_TABLE
        logger.info(
            f"DB engine created in {time.perf_counter()-t0:.2f}s "
            f"(host={DB_HOST}, db={DB_NAME}, driver={DB_DRIVER})"
        )

        # ORM-ish Core metadata for auth tables
        self.metadata = MetaData()
        self._define_auth_schema()
        self._ensure_auth_schema()

        # Quick smoke tests (no sensitive info logged)
        try:
            with self.engine.connect() as con:
                t = time.perf_counter()
                con.execute(text("SELECT 1"))
                logger.info(f"[smoke] SELECT 1 ok in {time.perf_counter()-t:.2f}s")
                t = time.perf_counter()
                cnt = con.execute(text(f"SELECT COUNT(*) FROM `{self.table}`")).scalar()
                logger.info(f"[smoke] {self.table} count={cnt} in {time.perf_counter()-t:.2f}s")
        except Exception:
            logger.exception("[smoke] DB connectivity test failed")

    # ============================================================
    #                    AUTH / ROLES / AUDIT
    # ============================================================
    def _define_auth_schema(self):
        self.users = Table(
            "users",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("username", String(100), nullable=False, unique=True),  # should match Salesperson
            Column("password_hash", String(255), nullable=False),
            Column("role", String(20), nullable=False, server_default=text("salesperson")),
            Column("is_active", Boolean, nullable=False, server_default=text("1")),
            Column("created_at", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
        )
        self.audit_log = Table(
            "audit_log",
            self.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=True),
            Column("username", String(100)),
            Column("action", String(255)),
            Column("extra", String(1000)),
            Column("ts", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
        )

    def _ensure_auth_schema(self):
        try:
            self.metadata.create_all(self.engine, checkfirst=True)
            logger.info("[auth] users & audit_log tables ensured")
        except Exception:
            logger.exception("[auth] ensure schema failed")

    @staticmethod
    def _norm_username(u: str) -> str:
        return (u or "").strip().upper()

    # ---------- User CRUD ----------
    def add_user(self, username: str, password: str, role: str = "salesperson", active: bool = True) -> bool:
        """(Legacy) Add user; prefer create_user. Kept for compatibility."""
        u = self._norm_username(username)
        pw_hash = pwd_context.hash(password)
        try:
            with self.engine.begin() as con:
                con.execute(
                    insert(self.users).values(
                        username=u,
                        password_hash=pw_hash,
                        role=role,
                        is_active=1 if active else 0,
                    )
                )
            logger.info(f"[users] added user {u} with role={role}")
            return True
        except Exception as e:
            logger.error(f"[users] add_user failed for {u}: {e}")
            return False

    def upsert_user(self, username: str, password: Optional[str] = None, role: Optional[str] = None, active: Optional[bool] = None) -> bool:
        """Create or update a user. If password is None, keep old hash."""
        u = self._norm_username(username)
        try:
            with self.engine.begin() as con:
                row = con.execute(
                    select(self.users.c.id, self.users.c.password_hash).where(self.users.c.username == u)
                ).fetchone()
                if row:
                    updates = {}
                    if password is not None:
                        updates["password_hash"] = pwd_context.hash(password)
                    if role is not None:
                        updates["role"] = role
                    if active is not None:
                        updates["is_active"] = 1 if active else 0
                    if updates:
                        con.execute(update(self.users).where(self.users.c.id == row.id).values(**updates))
                        logger.info(f"[users] updated user {u}: {list(updates.keys())}")
                else:
                    if password is None:
                        logger.error("[users] upsert_user requires password when creating a new user")
                        return False
                    con.execute(
                        insert(self.users).values(
                            username=u,
                            password_hash=pwd_context.hash(password),
                            role=role or "salesperson",
                            is_active=1 if (active is None or active) else 0,
                        )
                    )
                    logger.info(f"[users] created new user {u}")
            return True
        except Exception as e:
            logger.error(f"[users] upsert_user failed for {u}: {e}")
            return False

    def delete_user(self, username: str) -> bool:
        u = self._norm_username(username)
        try:
            with self.engine.begin() as con:
                con.execute(delete(self.users).where(self.users.c.username == u))
            logger.info(f"[users] deleted user {u}")
            return True
        except Exception as e:
            logger.error(f"[users] delete_user failed for {u}: {e}")
            return False

    def create_user(self, username: str, password: str, role: str = "salesperson") -> bool:
        """
        Create a new user with pbkdf2_sha256-hashed password.
        Returns True on success, False if username already exists.
        """
        u = self._norm_username(username)
        if not u or not password:
            raise ValueError("Username and password are required")

        try:
            with self.engine.begin() as con:
                exists = con.execute(
                    select(self.users.c.id).where(self.users.c.username == u)
                ).fetchone()
                if exists:
                    logger.warning(f"[users] User {u} already exists")
                    return False

                pw_hash = pwd_context.hash(password)
                values = {"username": u, "password_hash": pw_hash, "role": role, "is_active": 1}
                con.execute(insert(self.users).values(**values))

            logger.info(f"[users] Created user {u} with role {role}")
            return True
        except Exception:
            logger.exception("[users] create_user failed")
            return False

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[str]]:
        """Returns (ok, role). ok=False if bad creds or inactive."""
        u = self._norm_username(username)
        try:
            with self.engine.begin() as con:
                row = con.execute(
                    select(self.users.c.password_hash, self.users.c.role, self.users.c.is_active)
                    .where(self.users.c.username == u)
                ).fetchone()

                if not row:
                    return (False, None)

                pw_hash, role, is_active = row

                if isinstance(pw_hash, (bytes, bytearray)):
                    pw_hash = pw_hash.decode("utf-8", "ignore")

                if is_active in (0, False):
                    return (False, None)

                if not pwd_context.verify(password, pw_hash):
                    return (False, None)

                # Optional: transparently upgrade hashes if policy changed
                if pwd_context.needs_update(pw_hash):
                    new_hash = pwd_context.hash(password)
                    try:
                        con.execute(
                            update(self.users)
                            .where(self.users.c.username == u)
                            .values(password_hash=new_hash)
                        )
                    except Exception:
                        logger.warning("[users] could not upgrade password hash for %s", u)

                return (True, role)

        except Exception:
            logger.exception(f"[users] authenticate_user failed for {u}")
            return (False, None)

    def get_user_role(self, username: str) -> Optional[str]:
        u = self._norm_username(username)
        try:
            with self.engine.connect() as con:
                row = con.execute(select(self.users.c.role).where(self.users.c.username == u)).fetchone()
            return row[0] if row else None
        except Exception:
            logger.exception(f"[users] get_user_role failed for {u}")
            return None

    def list_users(self) -> List[Dict[str, Any]]:
        try:
            with self.engine.connect() as con:
                rows = con.execute(
                    select(self.users.c.username, self.users.c.role, self.users.c.is_active, self.users.c.created_at)
                    .order_by(self.users.c.username)
                ).fetchall()
            return [{"username": r.username, "role": r.role, "is_active": bool(r.is_active), "created_at": r.created_at} for r in rows]
        except Exception:
            logger.exception("[users] list_users failed")
            return []

    def write_audit(self, username: str, action: str, extra: Optional[str] = None) -> None:
        u = self._norm_username(username)
        try:
            with self.engine.begin() as con:
                con.execute(insert(self.audit_log).values(username=u, action=action, extra=(extra or "")[:1000]))
        except Exception:
            logger.exception(f"[audit] write failed for {u} action={action!r}")

    # Optional: bootstrap users from Salesperson names
    def ensure_users_for_distinct_salespeople(self, default_password: str = "changeme", default_role: str = "salesperson") -> int:
        """
        Create missing users for each distinct Salesperson in sales_report.
        Returns the number of users created.
        """
        try:
            ids, _months = self._discover_columns(self.table)
            lower_map = {c.lower(): c for c in ids}
            sp_col = (
                lower_map.get("salesman") or lower_map.get("salesperson") or lower_map.get("sales man name") or ids[1]
            )
            q = text(f"SELECT DISTINCT `{sp_col}` AS sp FROM `{self.table}` WHERE `{sp_col}` IS NOT NULL AND TRIM(`{sp_col}`) <> ''")
            with self.engine.connect() as con:
                sps = [r.sp for r in con.execute(q).fetchall()]
            created = 0
            with self.engine.begin() as con:
                existing = set(u.username for u in con.execute(select(self.users.c.username)).fetchall())
                for sp in sps:
                    u = self._norm_username(sp)
                    if u and u not in existing:
                        con.execute(
                            insert(self.users).values(
                                username=u,
                                password_hash=pwd_context.hash(default_password),
                                role=default_role,
                                is_active=1,
                            )
                        )
                        created += 1
                        logger.info(f"[bootstrap] created user for salesperson={u}")
            return created
        except Exception:
            logger.exception("[bootstrap] ensure_users_for_distinct_salespeople failed")
            return 0

    # ============================================================
    #                    SALES / BOOKINGS LOAD
    # ============================================================
    def load_sales_data(self) -> dict:
        logger.info("Starting load_sales_data()")
        try:
            ids, months = self._discover_columns(self.table)
            df = self._select_streaming(self.table, ids, months, label="Sales")
            aggs = self._aggregate_common(df, value_label="Sales")
            return {"raw": df if isinstance(df, pd.DataFrame) else pd.DataFrame(), **aggs}
        except Exception:
            logger.exception("[sales] load failed")
            raise

    def load_bookings_data(self) -> dict:
        logger.info("Starting load_bookings_data()")
        try:
            ids, months = self._discover_columns(DB_TABLE_BOOKINGS)
            df = self._select_streaming(DB_TABLE_BOOKINGS, ids, months, label="Bookings")
            aggs = self._aggregate_common(df, value_label="Bookings")
            return {"raw": df if isinstance(df, pd.DataFrame) else pd.DataFrame(), **aggs}
        except Exception:
            logger.exception("[bookings] load failed")
            raise

    # ---------------- Column discovery ----------------
    def _discover_columns(self, table: str) -> tuple[list[str], list[str]]:
        logger.info(f"[discover] Reading columns for `{table}` from INFORMATION_SCHEMA")
        q = text("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :db AND TABLE_NAME = :tbl
            ORDER BY ORDINAL_POSITION
        """)
        t0 = time.perf_counter()
        with self.engine.connect() as con:
            cols = [r[0] for r in con.execute(q, {"db": DB_NAME, "tbl": table}).all()]
        logger.info(f"[discover] {len(cols)} columns fetched in {time.perf_counter()-t0:.2f}s")

        cols_norm = [c.strip() if isinstance(c, str) else c for c in cols]
        lower_map = {c.lower(): c for c in cols_norm if isinstance(c, str)}

        name_col = (
            lower_map.get("customer")
            or lower_map.get("customername")
            or lower_map.get("customer name")
            or "Customer"
        )
        sales_col = (
            lower_map.get("salesman")
            or lower_map.get("salesperson")
            or lower_map.get("sales man name")
            or "Salesman"
        )
        if name_col not in cols_norm or sales_col not in cols_norm:
            raise ValueError(f"[discover] Missing required cols (Customer, Salesman). Got: {cols_norm[:10]}...")

        month_cols = []
        for c in cols_norm:
            if not isinstance(c, str):
                continue
            if any(tok in c.upper() for tok in EXCLUDE_TOKENS):
                continue
            if MONTH_HEADER_RE.match(c):
                month_cols.append(c)

        if not month_cols:
            raise ValueError("[discover] No month columns like '2024-Jan' found")

        preview = month_cols[:8]
        logger.info(f"[discover] ID cols: Customer={name_col!r}, Salesperson={sales_col!r}")
        logger.info(f"[discover] Month cols: {len(month_cols)} (preview {preview}{' …' if len(month_cols)>8 else ''})")

        return [name_col, sales_col], month_cols

    # ---------------- Streaming SELECT ----------------
    def _select_streaming(self, table: str, id_cols: list[str], month_cols: list[str], label: str) -> pd.DataFrame:
        safe_cols = [*id_cols, *month_cols]
        col_list = ", ".join([f"`{c}`" for c in safe_cols])
        sql = f"SELECT {col_list} FROM `{table}`"
        logger.info(f"[{label}] Executing streaming SELECT: {len(safe_cols)} cols from `{table}`")

        t0 = time.perf_counter()
        total = 0
        chunks = []
        try:
            with self.engine.connect().execution_options(stream_results=True) as con:
                for i, chunk in enumerate(pd.read_sql(sql, con, chunksize=10000)):
                    total += len(chunk)
                    chunks.append(chunk)
                    if i % 10 == 0:
                        logger.info(f"[{label}] Pulled {total} rows so far… ({time.perf_counter()-t0:.2f}s)")
        except Exception:
            logger.exception(f"[{label}] Streaming SELECT failed")
            raise

        if not chunks:
            logger.info(f"[{label}] SELECT returned 0 rows in {time.perf_counter()-t0:.2f}s")
            df = pd.DataFrame(columns=id_cols + month_cols)
        else:
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"[{label}] SELECT done: {total} rows in {time.perf_counter()-t0:.2f}s")

        return self._prepare_long(df, id_cols=id_cols, month_cols=month_cols, label=label)

    # ---------------- Longify & normalize ----------------
    def _prepare_long(self, df: pd.DataFrame, id_cols: list[str], month_cols: list[str], label: str) -> pd.DataFrame:
        t_long = time.perf_counter()
        df_use = df[id_cols + month_cols].copy()
        lower_map = {c.lower(): c for c in id_cols}
        name_col = lower_map.get("customer") or lower_map.get("customername") or lower_map.get("customer name") or id_cols[0]
        sales_col = lower_map.get("salesman") or lower_map.get("salesperson") or lower_map.get("sales man name") or id_cols[1]

        if name_col not in df_use.columns:
            name_col = id_cols[0]
        if sales_col not in df_use.columns:
            sales_col = id_cols[1]

        df_use.rename(columns={name_col: "CustomerName", sales_col: "Salesperson"}, inplace=True)

        melted = df_use.melt(id_vars=["CustomerName", "Salesperson"],
                             value_vars=month_cols,
                             var_name="MonthStr", value_name=label)
        melted[label] = pd.to_numeric(melted[label], errors="coerce").fillna(0)
        t_parse = time.perf_counter()
        melted["Month"] = pd.to_datetime(melted["MonthStr"], format="%Y-%b", errors="coerce")
        logger.info(f"[{label}] Month parse took {time.perf_counter()-t_parse:.2f}s")
        melted = melted.dropna(subset=["Month"])
        melted["Year"] = melted["Month"].dt.year.astype(int)

        logger.info(f"[{label}] Longify complete: shape={melted.shape} in {time.perf_counter()-t_long:.2f}s")
        return melted

    # ---------------- Aggregations (shared) ----------------
    def _aggregate_common(self, long_df: pd.DataFrame, value_label: str) -> dict:
        t_clip = time.perf_counter()
        long_df = long_df[long_df["Month"] <= CLIP_TO_MONTH].copy()
        logger.info(f"[{value_label}] Clip to <= {CLIP_TO_MONTH.date()} in {time.perf_counter()-t_clip:.2f}s")

        t1 = time.perf_counter()
        total_sc = (
            long_df.groupby(["Salesperson", "CustomerName"], as_index=False)[value_label].sum()
            .sort_values(["Salesperson", value_label], ascending=[True, False])
        )
        logger.info(f"[{value_label}] total_sc in {time.perf_counter()-t1:.2f}s (rows={total_sc.shape[0]})")

        t2 = time.perf_counter()
        yearly_sc = (
            long_df.groupby(["Salesperson", "CustomerName", "Year"], as_index=False)[value_label].sum()
            .sort_values(["Salesperson", "CustomerName", "Year"])
        )
        logger.info(f"[{value_label}] yearly_sc in {time.perf_counter()-t2:.2f}s (rows={yearly_sc.shape[0]})")

        t3 = time.perf_counter()
        monthly_sc = (
            long_df.groupby(["Salesperson", "CustomerName", "Month"], as_index=False)[value_label].sum()
            .sort_values(["Salesperson", "CustomerName", "Month"])
        )
        logger.info(f"[{value_label}] monthly_sc in {time.perf_counter()-t3:.2f}s (rows={monthly_sc.shape[0]})")

        years = sorted(long_df["Year"].unique())
        t4 = time.perf_counter()
        per_year_top20 = (
            yearly_sc.sort_values(["Salesperson", "Year", value_label], ascending=[True, True, False])
            .groupby(["Salesperson", "Year"], group_keys=False)
            .head(20)
            .reset_index(drop=True)
        )
        logger.info(f"[{value_label}] per_year_top20 in {time.perf_counter()-t4:.2f}s")

        return {
            "long": long_df,
            "total_sc": total_sc,
            "yearly_sc": yearly_sc,
            "monthly_sc": monthly_sc,
            "years": years,
            "per_year_top20": per_year_top20,
        }

    # ============================================================
    #              Convenience helpers for the app
    # ============================================================
    def filter_frames_by_user_role(
        self,
        sales_df: pd.DataFrame,
        bookings_df: pd.DataFrame,
        username: str,
        role: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        If role is salesperson -> lock to their own records.
        If role is manager/admin -> allow 'All' (no filter).
        Returns: (sales_filtered, bookings_filtered, selected_sp)
        """
        u = self._norm_username(username)
        role = (role or "").lower()
        if role in ("manager", "admin"):
            return sales_df, bookings_df, "All"
        # salesperson: filter to their name
        return (
            sales_df[sales_df["Salesperson"].str.upper() == u],
            bookings_df[bookings_df["Salesperson"].str.upper() == u],
            u,
        )
