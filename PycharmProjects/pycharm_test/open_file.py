import os

print(os.getcwd())

score_list = []

score_list_file = open("score", "r")

for score in score_list_file:
    # rstrip() 行末の文字（改行コード）を削除
    score = score.rstrip().split(",")
    score_list.append([score[0], int(score[1])])

score_list_file.close()
print(score_list_file.closed)

print(score_list)

score_list2 = []

with open("score", "r") as f:
    for score2 in f:
        score2 = score2.rstrip().split(",")
        score_list2.append([score2[0], int(score2[1])])

print(f.closed)
print(score_list2)
