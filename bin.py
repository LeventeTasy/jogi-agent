

name = f"sourceTRVK:p3434:5"

splitted = name.split(":")
source = splitted[0]
page = int(splitted[1][1:])
article = splitted[2]

print(source, page, article)