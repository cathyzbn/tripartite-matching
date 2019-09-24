f = open('output.txt', 'r')
string = f.readline()
while string:
    if string[:6] == "anneal":
        print(string[7:13])
    string = f.readline()
