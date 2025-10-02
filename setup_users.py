from conn import Conn

conn = Conn()

# Create a new salesperson user
ok = conn.create_user("CAROL", "Carol@123", role="salesperson")
ok1 = conn.create_user("Admin", "Cadmin123", role="admin")
if ok and ok1:
    print("✅ User created successfully")
else:
    print("⚠️ User already exists or failed")