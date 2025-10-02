#auth.py - Authentication module to handle login checks and permissions
import streamlit as st
from typing import Optional, List, Dict, Any
import time
import logging

# Set up logging
logger = logging.getLogger(__name__)

class Auth:
    """Authentication and authorization handler for Streamlit app"""
    
    # Define role-based page access
    ROLE_PERMISSIONS = {
        "admin": ["all"],  # Admin can see all pages
        "manager": ["all"],  # Manager can see all pages  
        "salesperson": ["Bookings", "Sales", "Dashboard", "Reports"],  # Limited access
        "viewer": ["Dashboard", "Reports"],  # Read-only access to reports
        "analyst": ["Bookings", "Targets", "Reports", "Dashboard"],  # Analyst role
    }
    
    # Define feature permissions
    FEATURE_PERMISSIONS = {
        "export_data": ["admin", "manager", "salesperson", "analyst"],
        "view_all_salespeople": ["admin", "manager", "analyst"],
        "use_ai_features": ["admin", "manager"],
        "modify_data": ["admin", "manager"],
        "view_sensitive_data": ["admin", "manager"],
        "access_targets": ["admin", "manager", "salesperson", "analyst"],
        "bulk_operations": ["admin", "manager"],
    }
    
    # Session timeout in minutes
    SESSION_TIMEOUT = 30
    
    @staticmethod
    def initialize_session():
        """Initialize session state variables if they don't exist"""
        if "logged_in" not in st.session_state:
            st.session_state.logged_in = False
        if "username" not in st.session_state:
            st.session_state.username = None
        if "role" not in st.session_state:
            st.session_state.role = None
        if "last_activity" not in st.session_state:
            st.session_state.last_activity = time.time()
    
    @staticmethod
    def check_authentication() -> bool:
        """Check if user is authenticated"""
        # Initialize session if needed
        Auth.initialize_session()
        
        if not st.session_state.get("logged_in", False):
            return False
        
        # Check session timeout
        if "last_activity" in st.session_state:
            elapsed = time.time() - st.session_state.last_activity
            if elapsed > Auth.SESSION_TIMEOUT * 60:
                logger.info(f"Session timeout for user {st.session_state.get('username', 'unknown')}")
                Auth.logout()
                return False
        
        # Update last activity
        st.session_state.last_activity = time.time()
        return True
    
    @staticmethod
    def get_current_user() -> Optional[str]:
        """Get current logged in username"""
        return st.session_state.get("username", None)
    
    @staticmethod
    def get_current_role() -> Optional[str]:
        """Get current user's role"""
        return st.session_state.get("role", None)
    
    @staticmethod
    def has_page_access(page_name: str) -> bool:
        """Check if current user has access to a specific page"""
        if not Auth.check_authentication():
            return False
        
        role = Auth.get_current_role()
        if not role:
            return False
        
        role_lower = role.lower()
        
        # Admin and manager have access to everything
        if role_lower in ["admin", "manager"]:
            return True
        
        # Check specific permissions
        allowed_pages = Auth.ROLE_PERMISSIONS.get(role_lower, [])
        if "all" in allowed_pages:
            return True
            
        # Check if page name matches any allowed page (case-insensitive)
        page_lower = page_name.lower()
        return any(page_lower in allowed.lower() for allowed in allowed_pages)
    
    @staticmethod
    def has_feature_permission(feature: str) -> bool:
        """Check if current user has permission for a specific feature"""
        if not Auth.check_authentication():
            return False
        
        role = Auth.get_current_role()
        if not role:
            return False
        
        role_lower = role.lower()
        allowed_roles = Auth.FEATURE_PERMISSIONS.get(feature, [])
        return role_lower in allowed_roles
    
    @staticmethod
    def require_auth(page_name: str = None):
        """
        Decorator/function to require authentication for a page
        Usage: Call at the top of each protected page
        """
        # Initialize session
        Auth.initialize_session()
        
        if not Auth.check_authentication():
            st.error("🔒 Please login to access this page")
            st.info("Navigate to the **Login** page to authenticate")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔐 Go to Login", type="primary", use_container_width=True):
                    st.switch_page("pages/00_Login.py")
            
            st.stop()
        
        # Check page-specific permissions if page_name provided
        if page_name and not Auth.has_page_access(page_name):
            st.error(f"⛔ Access Denied")
            st.warning(f"You don't have permission to view **{page_name}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"Your role: **{Auth.get_current_role()}**")
            with col2:
                st.info(f"Required: **Manager** or **Admin**")
            
            st.markdown("### 📋 Your Available Pages:")
            role = Auth.get_current_role()
            if role:
                allowed = Auth.ROLE_PERMISSIONS.get(role.lower(), [])
                if allowed:
                    for page in allowed:
                        if page != "all":
                            st.markdown(f"- {page}")
                else:
                    st.markdown("No pages available for your role")
            
            st.markdown("---")
            st.caption("Please contact your administrator if you believe you should have access to this page")
            st.stop()
    
    @staticmethod
    def logout():
        """Clear session and logout user"""
        # Get username before clearing for logging
        username = st.session_state.get("username", "unknown")
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Explicitly set logged_in to False
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None
        
        logger.info(f"User {username} logged out")
    
    @staticmethod
    def get_user_display_info() -> Dict[str, Any]:
        """Get user info for display"""
        return {
            "username": Auth.get_current_user(),
            "role": Auth.get_current_role(),
            "logged_in": Auth.check_authentication(),
            "can_export": Auth.has_feature_permission("export_data"),
            "can_view_all": Auth.has_feature_permission("view_all_salespeople"),
            "can_use_ai": Auth.has_feature_permission("use_ai_features"),
            "can_modify": Auth.has_feature_permission("modify_data"),
            "can_bulk_ops": Auth.has_feature_permission("bulk_operations"),
        }
    
    @staticmethod
    def show_user_sidebar():
        """Display user info in sidebar"""
        if Auth.check_authentication():
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 👤 User Info")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**User:**")
                    st.markdown(f"**Role:**")
                with col2:
                    st.markdown(f"{Auth.get_current_user()}")
                    st.markdown(f"{Auth.get_current_role()}")
                
                # Show permissions
                with st.expander("📋 My Permissions"):
                    info = Auth.get_user_display_info()
                    
                    # Data access
                    st.markdown("**Data Access:**")
                    if info["can_view_all"]:
                        st.success("✅ Can view all salespeople")
                    else:
                        st.info("👤 Can view own data only")
                    
                    # Export capability
                    st.markdown("**Export:**")
                    if info["can_export"]:
                        st.success("✅ Can export data")
                    else:
                        st.warning("⚠️ Cannot export data")
                    
                    # AI features
                    st.markdown("**AI Features:**")
                    if info["can_use_ai"]:
                        st.success("✅ AI features enabled")
                    else:
                        st.info("🤖 AI features disabled")
                    
                    # Data modification
                    if info["can_modify"]:
                        st.markdown("**Admin:**")
                        st.success("✅ Can modify data")
                
                # Session info
                with st.expander("⏱️ Session Info"):
                    if "last_activity" in st.session_state:
                        elapsed = int((time.time() - st.session_state.last_activity) / 60)
                        remaining = Auth.SESSION_TIMEOUT - elapsed
                        if remaining > 0:
                            st.info(f"Session timeout in: **{remaining}** min")
                        else:
                            st.warning("Session about to expire!")
                    st.caption(f"Timeout: {Auth.SESSION_TIMEOUT} minutes")
                
                # Logout button
                if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                    Auth.logout()
                    st.rerun()
                
                st.markdown("---")
    
    @staticmethod
    def get_role_description(role: str) -> str:
        """Get a description of what a role can do"""
        role_descriptions = {
            "admin": "Full system access - can view all data, manage users, and access all features",
            "manager": "Can view all salespeople data, export reports, and use AI features",
            "salesperson": "Can view own sales data, bookings, and basic reports",
            "viewer": "Read-only access to dashboards and reports",
            "analyst": "Can view all data, access analytics features, and export reports"
        }
        return role_descriptions.get(role.lower(), "Custom role with specific permissions")
    
    @staticmethod
    def check_password_strength(password: str) -> tuple[bool, str]:
        """
        Check if password meets minimum requirements
        Returns: (is_valid, message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        
        if not (has_upper and has_lower and has_digit):
            return False, "Password must contain uppercase, lowercase, and numbers"
        
        return True, "Password meets requirements"
    
    @staticmethod
    def format_last_login(timestamp) -> str:
        """Format a timestamp for display"""
        if timestamp:
            try:
                from datetime import datetime
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp)
                else:
                    dt = timestamp
                return dt.strftime("%Y-%m-%d %H:%M")
            except:
                return "Unknown"
        return "Never"
    
    @staticmethod
    def is_first_login() -> bool:
        """Check if this is the user's first login (to prompt password change)"""
        # This would typically check against a database flag
        # For now, return False as a placeholder
        return st.session_state.get("first_login", False)
    
    @staticmethod
    def requires_password_change() -> bool:
        """Check if user needs to change their password"""
        # Check if using default password pattern (username + "123")
        username = Auth.get_current_user()
        if username:
            # This is a simplified check - in production, you'd check against database
            return st.session_state.get("using_default_password", False)
        return False