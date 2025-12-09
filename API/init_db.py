from .database import Base, engine
from .models import User

def init():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init()
