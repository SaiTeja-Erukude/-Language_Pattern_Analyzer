import requests

def fetch_wikipedia_article(title):
    """Fetch the full plain text content of a Wikipedia article."""
    
    params = {
        'action': 'query',
        'format': 'json',
        'titles': title,
        'prop': 'extracts',
        'explaintext': True,
    }    
    response = requests.get('https://en.wikipedia.org/w/api.php', params=params).json()
    page     = next(iter(response['query']['pages'].values()))
    
    return page.get('extract', '')


def save_article_to_file(title, directory):
    """Save the fetched Wikipedia article content to a text file with a specific name format and return the word count."""
    
    content     = fetch_wikipedia_article(title)
    words       = content.split()
    word_count  = len(words)
    
     # Ensure the title is URL/formatted correctly
    formatted_title = title.replace(" ", "_")
    
    # Construct the file name and path
    filename = f"{formatted_title}_{word_count}_words.txt"
    filepath = f"{directory}/{filename}" 
    
    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(content)
    
    print(f'Article "{title}" saved to {filepath} with {word_count} words.')
    return word_count, filepath



if __name__ == '__main__':
    
    title     = "Macbeth"
    directory = ""

    word_count, filepath = save_article_to_file(title, directory)
    print(f"File saved: {filepath} with {word_count} words.")