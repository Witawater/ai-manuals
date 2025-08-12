# debug_key.py
from auth import require_api_key
from fastapi import Header

customer = require_api_key(X-API-Key="d4d6aa9f2bf75faf76454d8621af7c01")
print("Resolved customer:", customer)
