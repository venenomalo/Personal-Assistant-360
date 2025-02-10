import sys
import os

# Agregar "Generador_plantillas" al path
sys.path.append(os.path.join(os.path.dirname(__file__), "Generador_plantillas"))

from Generador_plantillas.app import app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)