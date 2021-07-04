import json

# for i in range(5):
#     with open(f"/Minions/submissions/test_ikhyo{i}.json", "r", encoding="utf-8-sig") as f:
#         colab_inference = json.load(f)
    
#     with open(f"/Minions/submissions/ikhyo{i}.json", "r", encoding="utf-8-sig") as f:
#         server_inference = json.load(f)

# for i in range(5):
#     with open(f"/Minions/check_folder/mi_{i}.json", "r", encoding="utf-8-sig") as f:
#         colab_inference = json.load(f)
    
#     with open(f"/Minions/submissions_origin/check_bertsum{i}.json", "r", encoding="utf-8-sig") as f:
#         server_inference = json.load(f)

# for i in range(5):
#     with open(f"/Minions/check_folder/ara_fold{i}.json", "r", encoding="utf-8-sig") as f:
#         colab_inference = json.load(f)
    
#     with open(f"/Minions/submissions_origin/check_kobert{i}.json", "r", encoding="utf-8-sig") as f:
#         server_inference = json.load(f)

with open(f"/Minions/submissions/test_bertsum0.json", "r", encoding="utf-8-sig") as f:
        colab_inference = json.load(f)
with open(f"/Minions/submissions/bertsum0.json", "r", encoding="utf-8-sig") as f:
        server_inference = json.load(f)

# with open(f"/Minions/submissions/test_sentavg.json", "r", encoding="utf-8-sig") as f:
#         colab_inference = json.load(f)
# with open(f"/Minions/submissions_origin/check_sentavg.json", "r", encoding="utf-8-sig") as f:
#         server_inference = json.load(f)

# with open(f"/Minions/submissions/kobert0.json", "r", encoding="utf-8-sig") as f:
#         colab_inference = json.load(f)
# with open(f"/Minions/submissions_origin/check_kobert0.json", "r", encoding="utf-8-sig") as f:
#         server_inference = json.load(f)

count = 0
dictionary = {0:0, 1:0, 2:0, 3:0}
for ids in range(len(colab_inference)):
    colab = set(colab_inference[ids].values())
    server = set(server_inference[ids].values())
    count = 0
    for colab_value in colab:
        if colab_value not in server:
            count +=1
    dictionary[count] += 1

print(dictionary)