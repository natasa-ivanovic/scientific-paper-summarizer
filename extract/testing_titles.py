import re


def check_numbered_list(lst):
    pattern = r'^\d+\.'
    last_number = 0

    for item in lst:
        match = re.match(pattern, item.strip())
        if not match:
            return False, "No numbering"
        number = int(match.group(0)[:-1])
        if number != last_number + 1 and number != last_number:
            return False, "Invalid order"
        last_number = number

    return True, None


def check_content(lst):
    contains_uvod = any("uvod" in c.lower() for c in lst)
    contains_zakljucak = any("zakljuc" in c.lower() or "zaključ" in c.lower() or "zakljuć" in c.lower() for c in lst)
    if not contains_uvod and not contains_zakljucak:
        return False, "No Uvod and Zakljucak"
    if not contains_uvod:
        return False, "No Uvod"
    if not contains_zakljucak:
        return False, "No Zakljucak"
    return True, None

elements = []

test = ['1. UVOD ', '2. KONFLIKTI U ORGANIZACIJI ', '3. PROBLEM ISTRAŽIVANJA ', '4. CILJ ISTRAŽIVANJA ', '5. NAČIN ISTRAŽIVANJA ', '6. HIPOTEZE ISTRAŽIVANJA ', ' 7. REZULTATI ISTRAŽIVANJA ', '8. ZAKLJUČAK']

with open("../titles.txt", 'r', encoding='utf-8') as file:
    for el in file:
        if not el.isspace():
            elements.append(el)

# print(elements)

cleaned_elements = []
for el in elements:
    file_name, titles = el.split("#&*")
    # print(file_name)
    # print(titles)
    file_name = file_name.split("/")[1]
    titles = titles.strip().split("|")
    obj = (file_name, titles)
    cleaned_elements.append(obj)

    # obj = (elements[i].split("/")[1], elements[i + 1].strip("\n").split("|"))
    # cleaned_elements.append(obj)

# print(cleaned_elements)

fail = 0
for key, value in cleaned_elements:
    val = [item.strip("''") for item in value]
    # ok, msg = check_numbered_list(val)
    ok, msg = check_content(val)
    if not ok:
        fail += 1
        print(key, ":", msg)
        print(val)
        print("-------------")
    # da li su svi numerisani
    # da li su numerisani kako treba
    # da li ima uvod
    # da li ima zakljucak

print("NUM INVALID: ", fail)
print("RESULT: ", (fail/len(cleaned_elements))*100)
