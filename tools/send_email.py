# import smtplib
# from email.mime.text import MIMEText
# #设置服务器所需信息
# #163邮箱服务器地址
# mail_host = 'smtp.163.com'
# #163用户名
# mail_user = 'castzhangzhengning'
# #密码(部分邮箱为授权码)
# mail_pass = 'UUURMGVZSFGNTERX'
# #邮件发送方邮箱地址
# sender = 'castzhangzhengning@163.com'
# #邮件接受方邮箱地址，注意需要[]包裹，这意味着你可以写多个邮件地址群发
# receivers = ['23880666@qq.com']
#
# #设置email信息
# #邮件内容设置
# message = MIMEText('content','plain','utf-8')
# #邮件主题
# message['Subject'] = 'title'
# #发送方信息
# message['From'] = sender
# #接受方信息
# message['To'] = receivers[0]
#
# #登录并发送邮件
# try:
#     smtpObj = smtplib.SMTP()
#     #连接到服务器
#     smtpObj.connect(mail_host,25)
#     #登录到服务器
#     smtpObj.login(mail_user,mail_pass)
#     #发送
#     smtpObj.sendmail(
#         sender,receivers,message.as_string())
#     #退出
#     smtpObj.quit()
#     print('success')
# except smtplib.SMTPException as e:
#     print('error',e) #打印错误

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
import sys




mailHost = 'smtp.163.com'
mailPort = 994

user = 'castzhangzhengning@163.com'
# password = os.environ.get("UUURMGVZSFGNTERX")
password = 'UUURMGVZSFGNTERX'

def send_mail(Subject='',message='',attach_file=''):
    receiver = ['23880666@qq.com', 'zzndream@gmail.com']# 收件人邮箱列表
    smtp = smtplib.SMTP_SSL(mailHost, mailPort)
    smtp.login(user=user, password=password)

    msg = MIMEMultipart()

    msg['Subject'] = Header(Subject + "训练完成", 'utf-8')
    msg['from'] = user
    msg['to'] = ','.join(receiver)

    # content = get_content()
    content = '当前网络已经训练完成，请前往查看结果\n'
    content = content + message
    msg_content = MIMEText(content, 'plain', 'utf-8')
    msg.attach(msg_content)

    if attach_file:
        att1 = MIMEText(open(attach_file, 'rb').read(), 'base64', 'utf-8')
        att1['Content-Type'] = 'application/octet-stream'
        att1['Content-Disposition'] = 'attachment; filename="report.txt"'
        msg.attach(att1)

    smtp.sendmail(user, receiver, msg.as_string())
    smtp.quit()
    print('success send mail')


if __name__ == '__main__':
    # file = '/home/zzn/Documents/zhangzhn_workspace/pycharm/mmdetection2.11/log/' \
    #        '20210528_cascade_mask_rcnn_r50_dilated_bifpn_1x_coco_train.txt'
    # print(sys.argv[0])          #sys.argv[0] 类似于shell中的$0,但不是脚本名称，而是脚本的路径
    # print(sys.argv[1])  # sys.argv[1] 表示传入的第一个参数，既 hello
    file = sys.argv[1]
    subject = (file.split("/",))[-1]
    send_mail(subject,'训练日志:\n',file)

# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
#
#
# def send_mail(title, message, receiver, attach_file=''):
#     # 建立连接
#     # python3.7版本开始，在SMTP建立阶段就要指明host地址，3.7之前不需要
#     s = smtplib.SMTP(host='host')
#     s.connect(host='host', port='port')
#
#     # 网站需要安全认证时添加
#     s.starttls()
#
#     # 登录发送邮件账户
#     s.login('username', 'password')
#
#     msg = MIMEMultipart()
#
#     content = MIMEText(message, 'html', 'utf-8')
#     msg['Subject'] = f'{title}'
#     msg['From'] = 'sender'
#     msg['To'] = receiver
#     msg.attach(content)
#
#     if attach_file:
#         att1 = MIMEText(open(attach_file, 'rb').read(), 'base64', 'utf-8')
#         att1['Content-Type'] = 'application/octet-stream'
#         att1['Content-Disposition'] = 'attachment; filename="report.txt"'
#         msg.attach(att1)
#
#     try:
#         s.send_message(msg)
#         status = 'Success'
#         print(status)
#     except smtplib.SMTPException as e:
#         print(e)
#         status = 'Failed'
#
#     s.quit()
#
#     return status
#
#
# if __name__ == '__main__':
#     send_mail('title', 'msg', 'receiver', 'attachment')
