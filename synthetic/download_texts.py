import requests

if __name__ == "__main__":

    for index in range(100):
        print("helo")
        response = requests.get('https://en.wikipedia.org/api/rest_v1/page/random/summary').json()
        print(response)
        text = response['extract'].strip()
        if len(text) > 32:
            with open('texts/%d.txt' % index, 'wt') as f:
                f.write(text)
