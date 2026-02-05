from  datetime import datetime
import uuid

def get_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_id():
    return str(uuid.uuid4())