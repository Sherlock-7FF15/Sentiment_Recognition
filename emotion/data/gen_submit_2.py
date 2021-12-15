import csv
r = open('chn_test.csv','r')
csvFile = open("gen_submit.csv", 'w', newline='')
writer = csv.writer(csvFile)
writer.writerow(('id', 'y'))
count = 0
for line in r:
    if count > 0:
        weiboId, weiboTime, userid, text, img, video, label = next(csv.reader(line.splitlines(),skipinitialspace=True))
        writer.writerow((weiboId.strip(),label))
    count += 1