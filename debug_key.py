# debug_key.py
from auth import require_api_key
from fastapi import Header

customer = require_api_key(x_api_key="d4d6aa9f2bf75faf76454d8621af7c01")
print("Resolved customer:", customer)
