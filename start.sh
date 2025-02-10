#!/bin/bash
gunicorn -w 2 -b 0.0.0.0:5000 --timeout 120 Generador_plantillas.app:app
