import sqlite3

def clear_datasets():
    """Borra todos los registros de la tabla datasets en la base de datos."""
    try:
        # Conectar a la base de datos
        conn = sqlite3.connect('./data/processed_datasets.db')
        cursor = conn.cursor()

        # Confirmación antes de eliminar
        confirm = input("¿Estás seguro de que quieres borrar todos los registros? (s/n): ")
        if confirm.lower() != 's':
            print("Operación cancelada.")
            return

        # Ejecutar comando para borrar todos los registros
        cursor.execute("DELETE FROM datasets")
        conn.commit()

        # Mostrar el número de registros eliminados
        print(f"Se han eliminado {cursor.rowcount} registros de la tabla datasets.")

        # Cerrar la conexión
        conn.close()
    except sqlite3.Error as e:
        print(f"Error al borrar los registros: {e}")

if __name__ == "__main__":
    clear_datasets()
