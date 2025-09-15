import pandas as pd
import json

def preprocess_json(players, uploaded_file, club_name):
    # Leere Liste zur Speicherung aller Kommentare
    all_comments = []

    # Lesen des Inhalts der hochgeladenen Datei
    try:
        # Falls die Datei Bytes enthält, dekodieren wir sie zu einem String
        json_str = uploaded_file.read().decode('utf-8')
        posts = json.loads(json_str)
    except Exception as e:
        return e

    # Überprüfen, ob 'posts' eine Liste von Posts ist
    if isinstance(posts, list):
        # Extrahiere die Kommentare aus jedem Post
        for post in posts:
            if 'comments' in post:
                for comment in post['comments']:
                    all_comments.append({
                        'post': post.get('link', ''),
                        'caption': post.get('caption', ''),
                        'comment': comment,
                    })
    elif isinstance(posts, dict) and 'comments' in posts:
        # Falls 'posts' ein einzelner Post ist
        for comment in posts['comments']:
            all_comments.append({
                'post': posts.get('link', ''),
                'caption': posts.get('caption', ''),
                'comment': comment,
            })
    else:
        st.warning("Die JSON-Datei enthält keine gültigen Posts oder Kommentare.")
        return pd.DataFrame()

    # Speichern aller Kommentare in einem Pandas DataFrame
    df = pd.DataFrame(all_comments)
    df = df[~df['comment'].str.contains("Original-Audio", case=False, na=False)]
    df = df[~df['comment'].str.contains("•", case=False, na=False)]
    df = df.drop_duplicates(subset=['comment'])

    results = []

    for player_name in players:
        # Bereinigung des Spielernamens
        player_name = player_name.strip()
        
        # Aufteilen des Namens in Teile
        name_parts = player_name.split()
        # Erstellen von Varianten des Namens
        variants = set()
        variants.add(player_name.lower())  # Voller Name

        # Nachname(n)
        last_name = name_parts[-1].lower()
        variants.add(last_name)
        print(f"Varianten für {player_name}: {variants}")

        # Filterung der Kommentare, die die Varianten enthalten
        mask = df['comment'].str.contains('|'.join(variants), case=False, na=False)
        matching_comments = df[mask]

        for index, row in matching_comments.iterrows():
            # Hinzufügen der Ergebnisse zur Liste
            results.append({
                'club': club_name,
                'player': player_name,
                'comment': row['comment'],
                'post': row['post'],
                'caption': row['caption']
            })

    df_players = pd.DataFrame(results)

    return df_players