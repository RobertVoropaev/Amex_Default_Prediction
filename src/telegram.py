import requests
import os

def telegram(message):
    project_dir = "/".join(os.getcwd().split("/")[3:])
    message = project_dir + ": " + message
    
    bot_token = '5477837037:AAHQC6CSEDgQVUkLJgTkkB1az6b0DcurgB4'
    bot_chatID = '304750473'
    send_text = (
        'https://api.telegram.org/bot' + bot_token + 
        '/sendMessage?chat_id=' + bot_chatID + 
        '&parse_mode=Markdown&text=' + message
    )

    response = requests.get(send_text)