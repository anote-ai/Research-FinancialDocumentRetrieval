import os
search = 'ContextualCompressionRetriever'
base = r'C:\Users\james\anaconda3\envs\findocretrieval\Lib\site-packages'
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                with open(path, encoding='utf-8', errors='ignore') as fp:
                    if search in fp.read():
                        print(path)
            except:
                pass