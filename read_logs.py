import pandas as pd

def read_logs(path):
    df = pd.read_json(path, lines=True)
    subjects = pd.DataFrame(df.skills.tolist(),
                            columns=[f'subject_{i+1}' for i in range(len(df.skills[0]))])
    return pd.concat((df, subjects), axis=1).drop(columns=['skills'])

df = read_logs('TRPO_application.log')
df = df[['action', 'subject', 'difficulty']]
df['difficulty_name'] = df.loc[df.difficulty == 1, 'difficulty'] = 'easy'
df['difficulty_name'] = df.loc[df.difficulty == 2, 'difficulty'] = 'medium'
df['difficulty_name'] = df.loc[df.difficulty == 3, 'difficulty'] = 'advanced'
df['subject_name'] = [str(x) for x in df['subject']]

df['name'] = df['action']+' in subject '+df['subject_name']+' with '+df['difficulty_name']+' difficulty'
previous = df.iloc[:-1, -1]
next = df.iloc[1:, -1]
previous.name = 'previous'
next.name= 'next'

df_flow = pd.DataFrame(zip(previous, next), columns=[previous.name, next.name])
count_series = df_flow.groupby(['previous', 'next']).size().reset_index()
print(42)