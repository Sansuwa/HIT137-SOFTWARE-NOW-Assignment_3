from gui import AppGUI
from models import ModelManager
from oop_demo import OOPDemo

def main():
    # Initialize model manager (lazy loads models on demand)
    model_manager = ModelManager()
    
    app.run()

if __name__ == '__main__':
    main()
