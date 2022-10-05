import os
import sys
from roman import fromRoman
from vietnam_number import n2w, n2w_single
from num2words import num2words

def remove_adjacent_word(text, word):
    text = text.strip()
    new_text = []
    tokens = [i.strip() for i in text.split()]
    if len(tokens) != 0:
        for i in range(len(tokens)-1):
            if tokens[i] == word and tokens[i+1] == word:
                continue
            else:
                new_text.append(tokens[i])
        new_text.append(tokens[-1])

    return ' '.join(new_text)

def remove_redundant_words(text, word1, word2):
    text = text.strip()
    new_text = []
    tokens = [i.strip() for i in text.split()]
    if len(tokens) != 0:
        for i in range(len(tokens)-1):
            if tokens[i] == word1 and tokens[i+1] == word2:
                continue
            else:
                new_text.append(tokens[i+1])
    new_text.insert(0, tokens[0])
    return ' '.join(new_text)


def remove_adjacent_dot(text):
    text = text.strip()
    new_text = []
    if len(text) != 0:
        for i in range(len(text)-1):
            if text[i] == '.' and text[i+1] == '.':
                continue
            else:
                new_text.append(text[i])
        new_text.append(text[-1])

    return ' '.join(new_text)

def remove_punct_date(text):
    punct = '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split()
    for e in punct:
        e = e.strip()
        text = text.replace(e, ' ')
    return text

def unit2words(input_str):
    # Units of speed	
    input_str = input_str.replace("bpm", " nhịp trên phút ")	
    input_str = input_str.replace("nm/s", " na nô mét trên giây ")	
    input_str = input_str.replace("µm/s", " mi cờ rô mét trên giây ")	
    input_str = input_str.replace("mm/s", " mi li mét trên giây ")	
    input_str = input_str.replace("cm/s", " xen ti mét trên giây ")	
    input_str = input_str.replace("dm/s", " đề xi mét trên giây ")	
    input_str = input_str.replace("dam/s", " đề ca mét trên giây ")	
    input_str = input_str.replace("hm/s", " héc tô mét trên giây ")	
    input_str = input_str.replace("km/s", " ki lô mét trên giây ")	
    input_str = input_str.replace("m/s", " mét trên giây ")	
    input_str = input_str.replace("nm/giây", " na nô mét trên giây ")	
    input_str = input_str.replace("µm/giây", " mi cờ rô mét trên giây ")	
    input_str = input_str.replace("mm/giây", " mi li mét trên giây ")	
    input_str = input_str.replace("cm/giây", " xen ti mét trên giây ")	
    input_str = input_str.replace("dm/giây", " đề xi mét trên giây ")	
    input_str = input_str.replace("dam/giây", " đề ca mét trên giây ")	
    input_str = input_str.replace("hm/giây", " héc tô mét trên giây ")	
    input_str = input_str.replace("km/giây", " ki lô mét trên giây ")	
    input_str = input_str.replace("m/giây", " mét trên giây ")	
    input_str = input_str.replace("nm/h", " na nô mét trên giờ ")	
    input_str = input_str.replace("µm/h", " mi cờ rô mét trên giờ ")	
    input_str = input_str.replace("mm/h", " mi li mét trên giờ ")	
    input_str = input_str.replace("cm/h", " xen ti mét trên giờ ")	
    input_str = input_str.replace("dm/h", " đề xi mét trên giờ ")	
    input_str = input_str.replace("dam/h", " đề ca mét trên giờ ")	
    input_str = input_str.replace("hm/h", " héc tô mét trên giờ ")	
    input_str = input_str.replace("km/h", " ki lô mét trên giờ ")	
    input_str = input_str.replace("kmh", " ki lô mét trên giờ ")	
    input_str = input_str.replace("m/h", " mét trên giờ ")	
    input_str = input_str.replace("nm/giờ", " na nô mét trên giờ ")	
    input_str = input_str.replace("µm/giờ", " mi cờ rô mét trên giờ ")	
    input_str = input_str.replace("mm/giờ", " mi li mét trên giờ ")	
    input_str = input_str.replace("cm/giờ", " xen ti mét trên giờ ")	
    input_str = input_str.replace("dm/giờ", " đề xi mét trên giờ ")	
    input_str = input_str.replace("dam/giờ", " đề ca mét trên giờ ")	
    input_str = input_str.replace("hm/giờ", " héc tô mét trên giờ ")	
    input_str = input_str.replace("km/giờ", " ki lô mét trên giờ ")	
    input_str = input_str.replace("m/giờ", " mét trên giờ ")	
    input_str = input_str.replace("m3/h", " mét khối trên giờ ")	
    # print(input_str)	
    # Others	
    input_str = input_str.replace("mAh", " mi li am pe giờ ")  	
    input_str = input_str.replace("kWh/ngày", " ki lô goát giờ trên ngày ")	
    input_str = input_str.replace("/tấn", " trên tấn ")	
    input_str = input_str.replace("/thùng", " trên thùng ")	
    input_str = input_str.replace("/căn", " trên căn ")	
    input_str = input_str.replace("/cái", " trên cái ")	
    input_str = input_str.replace("/con", " trên con ")	
    input_str = input_str.replace("/lần", " trên lần ")	
    input_str = input_str.replace("/năm", " trên năm ")	
    input_str = input_str.replace("/tháng", " trên tháng ")	
    input_str = input_str.replace("/ngày", " trên ngày ")	
    input_str = input_str.replace("/giờ", " trên giờ ")	
    input_str = input_str.replace("/phút", " trên phút ")	
    input_str = input_str.replace("đ/CP", " đồng trên cổ phiếu ")	
    input_str = input_str.replace("đ/lít", " đồng trên lít ")	
    input_str = input_str.replace("đ/lượt", " đồng trên lượt ")	
    input_str = input_str.replace("người/", " người trên ")	
    input_str = input_str.replace("giờ/", " giờ trên ")	
    input_str = input_str.replace('%', ' phần trăm ')	
    input_str = input_str.replace('mAh ', ' mi li am pe ')	
    input_str = input_str.replace(" lít/", " lít trên ")	
    input_str = input_str.replace("./", "")	
    input_str = input_str.replace("Nm", " Niu tơn mét ")	
    input_str = input_str.replace("º", " độ ")	
    input_str = input_str.replace("vòng 1/", " vòng 1 ")	
    input_str = input_str.replace("mmol/l", " mi li mon trên lít ")	
    input_str = input_str.replace("mg/", " mi li gam trên ")	
    # input_str = input_str.replace("g/", " gam trên ")	
    input_str = input_str.replace("kg/", " ki lô gam trên ")	
    input_str = input_str.replace("cái/", " cái trên ")	
    input_str = input_str.replace("triệu/", " triệu trên ")	
    input_str = input_str.replace("g/km", " gam trên ki lô mét ")	
    input_str = input_str.replace("ounce", " ao ")	
    input_str = input_str.replace("m3/s", " mét khối trên giây ")

    # Units of information
    input_str = input_str.replace(' byte ', ' bai ')
    input_str = input_str.replace(' KB ', ' ki lô bai ')
    input_str = input_str.replace(' Mb ', ' mê ga bai ')
    input_str = input_str.replace(' MB ', ' mê ga bai ')
    input_str = input_str.replace(' Gb ', ' ghi ga bai ')
    input_str = input_str.replace(' GB ', ' ghi ga bai ')
    input_str = input_str.replace(' gb ', ' ghi ga bai ')
    input_str = input_str.replace(' TB ', ' tê ra bai ')
    # print(input_str)
    # Unit of volume
    input_str = input_str.replace('dm3', ' đề xi mét khối ')
    input_str = input_str.replace('cm3', ' xen ti mét khối ')
    input_str = input_str.replace('m3', ' mét khối ')
    # input_str = input_str.replace(' 2G ', ' hai gờ ')
    # input_str = input_str.replace(' 3G ', ' ba gờ ')
    # input_str = input_str.replace(' 4G ', ' bốn gờ ')
    # input_str = input_str.replace(' 5G ', ' năm gờ ')
    # print(input_str)

    # Units of frequency
    input_str = input_str.replace('kHz', ' ki lô héc ')	
    input_str = input_str.replace('GHz', ' ghi ga héc ')	
    input_str = input_str.replace('MHz', ' mê ga héc ')	
    input_str = input_str.replace('Hz', ' héc ')

    # Units of data-rate
    input_str = input_str.replace('Mbps', ' mê ga bít trên giây ')
    input_str = input_str.replace('Mb/s', ' mê ga bít trên giây ')
    
    # Units of currency
    input_str = input_str.replace("đồng/", " đồng trên ")
    input_str = input_str.replace("USD/", " u ét đê trên ")
    input_str = input_str.replace(' đ ', ' đồng ')
    input_str = input_str.replace('$', ' đô la ')
    input_str = input_str.replace('USD', ' u ét đê ')
    input_str = input_str.replace('Euro', ' ơ rô ')
    input_str = input_str.replace('VNĐ', ' việt nam đồng ')
    input_str = input_str.replace('vnđ', ' việt nam đồng ')
    # input_str = input_str.replace('vnd', ' đồng ')
    # input_str = input_str.replace('VND', ' đồng ')

    # Units of area
    input_str = input_str.replace('km2', ' ki lô mét vuông ')
    input_str = input_str.replace('Km2', ' ki lô mét vuông ')
    input_str = input_str.replace('cm2', ' xen ti mét vuông ')
    input_str = input_str.replace('mm2', ' mi li mét vuông ')
    input_str = input_str.replace('m2', ' mét vuông ')
    input_str = input_str.replace(' ha ', ' héc ta ')
    input_str = input_str.replace('hecta', ' héc ta ')
    
    # Units of length
    input_str = input_str.replace(' km ', ' ki lô mét ')
    input_str = input_str.replace(' cm ', ' xen ti mét ')
    input_str = input_str.replace(' mm ', ' mi li mét ')
    input_str = input_str.replace(' nm ', ' na nô mét ')
    input_str = input_str.replace(' inch ', ' inh ')
    
    # Units of volume
    input_str = input_str.replace(' ml ', ' mi li lít ')
    input_str = input_str.replace(' gr ', ' gam ')
    input_str = input_str.replace(' dm3 ', ' đề xi mét khối ')
    input_str = input_str.replace(' cm3 ', ' xen ti mét khối ')
    input_str = input_str.replace(' cc ', ' xê xê ')	
    input_str = input_str.replace(' m3 ', ' mét khối ')	



    # Units of weight
    input_str = input_str.replace('/kg', ' trên một ki lô gam ')
    input_str = input_str.replace('kg/', ' ki lô gam trên ')
    input_str = input_str.replace(' kg ', ' ki lô gam ')
    input_str = input_str.replace(' Kg ', ' ki lô gam ')
    input_str = input_str.replace(' grams ', ' gờ ram ')
    input_str = input_str.replace(' mg ', ' mi li gam ')

    input_str = input_str.replace(' kW ', ' ki lô goát ')
    

    # Units of temperature
    input_str = input_str.replace("oC", " độ xê ")
    input_str = input_str.replace("°C", " độ xê ")
    input_str = input_str.replace("ºC", " độ xê ")
    input_str = input_str.replace("ºF", " độ ép ")
    input_str = input_str.replace("oF", " độ ép ")
    
    # Picture element
    input_str = input_str.replace(' MP ', ' mê ga píc xeo ')

    return input_str

def date_dmy2words(date):
    # print(date)
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if '/' in date:
        day, month, year = date.split("/") 
    elif '.' in date:
        day, month, year = date.split(".") 
    elif '-' in date:
        day, month, year = date.split("-") 

    date_str = ' ngày ' + num2words_fixed(str(int(day))) + ' tháng ' + num2words_fixed(str(int(month))) + ' năm ' +  num2words_fixed(year)
    return date_str

def math_characters(input_str):
    input_str = input_str.replace('²', ' bình phương ')
    input_str = input_str.replace('π', ' pi ')
    return input_str

def date_dm2words(date):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    if '/' in date:
        day, month = date.split("/")
    elif '-' in date:
        day, month = date.split("-")
    else:
        day, month = date.split(".")
    date_str = ' ngày ' + num2words_fixed(str(int(day))) + ' tháng ' + num2words_fixed(str(int(month)))
    return date_str

def date_my2words(date):
    if date[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        date = date[:-1]
    # print(date)
    if '/' in date:
        month, year = date.split("/") 
    elif '.' in date:
        month, year = date.split(".") 
    elif '-' in date:
        month, year = date.split("-") 

    date_str = ' tháng ' + num2words_fixed(str(int(month))) + ' năm ' +  num2words_fixed(year)
    return date_str

def num2words_fixed(num):
    if num[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        num = num[:-1]

    if int(num) < 1000000:
        if int(num) == 0:
            num_str = n2w(str(int(num)))
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")
            if int(num) % 1000 == 0:
                num_str = n2w(num).replace("không trăm", "")

    elif int(num) >=  1000000 and int(num) < 1000000000:
        if int(num) % 1000 == 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm nghìn', '')[:-10]
        elif int(num) % 1000 != 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm nghìn', '')
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")

    elif int(num) >=  1000000000 and int(num) < 1000000000000:
        if int(num) % 1000 == 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm triệu', '').replace('không trăm nghìn', '')[:-10]
        elif int(num) % 1000 != 0:
            num_str = n2w(num).replace("lẽ", "lẻ").replace('không trăm triệu', '').replace('không trăm nghìn', '')
        else:
            num_str = n2w(num).replace("lẽ", "lẻ")

    return num_str

def time2words(time):
    if time[-1] == 's':
        time = time[:-1]
    if ":" in time:
        time_split = time.split(":")
        if len(time_split) == 2:
            hour, minute = time_split
            time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút "

        elif len(time_split) == 3:
            hour, minute, second = time_split
            time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút " + num2words_fixed(str(int(second))) + " giây "
    elif "h" in time:
        hour, minute = time.split("h")
        if minute != "":
            if 'p' in minute:
                minute, second = minute.split('p')
                time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút " + num2words_fixed(str(int(second))) + " giây "
            else:
                time_str = num2words_fixed(str(int(hour))) + ' giờ ' +  num2words_fixed(str(int(minute))) + " phút "
        else:
            time_str = num2words_fixed(str(int(hour))) + ' giờ '
    time_str = time_str.replace('không phút', '')
    return time_str

def multiply(input_str):
    element_split = input_str.split("x")
    multiply_str_list = []
    for element in element_split:
        multiply_str_list.append(' ' + n2w(str(int(element))) + ' ')
    multiply_str = ' nhân '.join(multiply_str_list)
    return multiply_str

def phone2words(number):
    # print(number)
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    number = number.replace(' ', '')
    return " " + n2w_single(number) + " "

def num2words_float(number):
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    interger, decimal = number.split(',')
    return num2words_fixed(interger) + ' phẩy ' + num2words_fixed(decimal)

def version2words(number):
    if number[-1] in '! " “ \' ( ) , . : ; ? [ ] _ ` { | } ~ … — 》 ‘ ’'.split():
        number = number[:-1]
    interger, decimal = number.split('.')
    return num2words_fixed(interger) + ' chấm ' + phone2words(decimal)