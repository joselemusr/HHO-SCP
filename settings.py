import os

package = "numpy"
try:
    import numpy
except:
    print(f"--------------- INSTALANDO {package} --------------------------------")
    os.system("pip install -U "+ package)

package = "python-dotenv"
try:
    from dotenv import load_dotenv
except:
    print(f"--------------- INSTALANDO {package} --------------------------------")

    os.system("pip install -U "+ package)

from dotenv import load_dotenv
load_dotenv()


package = "envs"
try:
    import envs as env
except:
    print(f"--------------- INSTALANDO {package} --------------------------------")
    os.system("pip install "+ package)


# package = "pathlib"
# try:
#     import pathlib
# except:
#     print(f"--------------- INSTALANDO {package} --------------------------------")
#     os.system("pip install "+ package)

package = "SQLAlchemy"
try:
    import sqlalchemy as db
except:
    print(f"--------------- INSTALANDO {package} --------------------------------")
    os.system("pip install "+ package)

package = "psycopg2-binary"
try:
    import psycopg2
except:
    print(f"--------------- INSTALANDO {package} --------------------------------")
    os.system("pip install "+ package)


