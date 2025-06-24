from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# Autoryzacja (przy pierwszym uruchomieniu otworzy przeglądarkę)
gauth = GoogleAuth()
gauth.LoadCredentialsFile("credentials.json")  # Ścieżka do pliku credentials.json

if gauth.credentials is None:
    # Jeśli nie mamy zapisanych danych logowania, to musimy przeprowadzić proces autoryzacji
    gauth.LocalWebserverAuth()  # Ta funkcja otworzy przeglądarkę, aby użytkownik mógł się zalogować
elif gauth.access_token_expired:
    # Jeśli token wygasł, to odświeżamy go
    gauth.Refresh()
else:
    # Jeśli mamy zapisane dane logowania, to ładujemy je
    gauth.Authorize()

# Stworzenie obiektu GoogleDrive
drive = GoogleDrive(gauth)

# Przesyłanie pliku
file = drive.CreateFile({'title': 'testowy_plik.txt'})
file.SetContentString('To jest testowy plik wrzucony z Pythona!')
file.Upload()
print(f"✅ Plik wrzucony! ID: {file['id']}")