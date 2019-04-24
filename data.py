import re
import pandas as pd


if __name__ == "__main__":
    pattern = re.compile(r'''<doc id=(\d+)>
    <summary>
        (.+)
    </summary>
    <short_text>
        (.+)
    </short_text>
</doc>''', re.M)

    with open('data/raw_data/DATA/PART_I.txt', encoding='utf-8') as f:
        text = ''.join(f.readlines())
    matches = re.findall(pattern, text)[::100]
    print(len(matches))
    df = pd.DataFrame(matches)
    df = df[:32]
    df.to_csv('data/processed_data/eval.csv', sep='\t', header=False, index=False)

    