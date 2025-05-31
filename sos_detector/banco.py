import psycopg2
import datetime
import geocoder

def conectar_banco():
    return psycopg2.connect(
        dbname="HELIOS", 
        user="natifss", 
        password="natalicristine10", 
        host="localhost"
    )

def salvar_alerta(caminho_video):
    conn = conectar_banco()
    cursor = conn.cursor()

    with open(caminho_video, "rb") as f:
        bin_video = f.read()

    cidade = "Desconhecida"
    g = geocoder.ip("me")
    if g.ok:
        cidade = g.city

    cursor.execute(
        "INSERT INTO alertas (data_hora, cidade, video) VALUES (%s, %s, %s)",
        (datetime.datetime.now(), cidade, psycopg2.Binary(bin_video))
    )
    conn.commit()
    conn.close()
