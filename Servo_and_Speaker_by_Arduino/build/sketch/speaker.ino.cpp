#include <Arduino.h>
#line 1 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\speaker\\speaker.ino"
/*********************************************************************
  文件名：zwvoice.ino
  描述：播放中文语音
  适用主板：Arduino UNO、NANO、Pro MINI、Leonado、MEGA2560或其他兼容板
  适用模块：VoicePro TTS语音合成模块
  配合模块：
  汉字网址：https://www.107000.com/T-Hex/
*********************************************************************/

#include <SoftwareSerial.h>

SoftwareSerial mySerial(22, 23); //使用软件串口，模块TX接D10，RX接D11
char Chinese_welcome[50] = {char(0xBB), char(0xB6), char(0xD3), char(0xAD), char(0xCA), char(0xB9), char(0xD3), char(0xC3), char(0xD6), char(0xC7), char(0xC4), char(0xDC), char(0xBE), char(0xFC), char(0xC6), char(0xEC), char(0xD6), char(0xFA), char(0xCA), char(0xD6)}; //欢迎使用智能军旗助手
void SYN_FrameInfo(uint8_t Music, char *HZdata);

#line 16 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\speaker\\speaker.ino"
void setup();
#line 28 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\speaker\\speaker.ino"
void loop();
#line 16 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\speaker\\speaker.ino"
void setup()
{
    Serial.begin(9600); //设置串口波特率9600
    Serial.println("Serial ON");
    mySerial.begin(9600);
    mySerial.print("<S>2"); //设置语速2，（1-3级可调）
    delay(50);
    mySerial.print("<V>3"); //设置音量3，（1-6级可调，声音过大会失真）
    delay(50);
    SYN_FrameInfo(0, Chinese_welcome); //播放“欢迎使用智能军旗助手”
}

void loop()
{
    //串口数据读取
    if(Serial.available())
    {
       inputChar = Serial.read() //char(Serial.read())
       if (inputChar=='0'){
         SYN_FrameInfo(0, Chinese_gw); //播放“你好我的朋友”
         Serial.println("green_win");
       }else if(inputChar=='1'){
         SYN_FrameInfo(0, Chinese_rw); //播放“你好我的朋友”
         Serial.println("red_win");
       }else if(inputChar=='2'){
         SYN_FrameInfo(0, Chinese_es); //播放“你好我的朋友”
         Serial.println("equal_siling");
       }else if(inputChar=='3'){
         SYN_FrameInfo(0, Chinese_rskbz); //播放“你好我的朋友”
         Serial.println("red_siling_killed_by_zhadan");
       }else if(inputChar=='4'){
         SYN_FrameInfo(0, Chinese_gskbz); //播放“你好我的朋友”
         Serial.println("green_siling_killed_by_zhadan");
       }else if(inputChar=='5'){
         SYN_FrameInfo(0, Chinese_equal); //播放“你好我的朋友”
         Serial.println("equal");
       }else if(inputChar=='6'){
         SYN_FrameInfo(0, Chinese_rskbz); //播放“你好我的朋友”
         Serial.println("red_siling_killed_by_dilei");
       }else if(inputChar=='7'){
         SYN_FrameInfo(0, Chinese_gskbz); //播放“你好我的朋友”
         Serial.println("green_siling_killed_by_dilei");
       }else if(inputChar=='8'){
         SYN_FrameInfo(0, Chinese_rkg); //播放“你好我的朋友”
         Serial.println("red_kill_green");
       }else if(inputChar=='9'){
        SYN_FrameInfo(0, Chinese_gkr); //播放“你好我的朋友”
         Serial.println("green_kill_red");
       }else if(inputChar=='A'){
         SYN_FrameInfo(0, Chinese_rkg); //播放“你好我的朋友”
         Serial.println("red_kill_green");
       }else if(inputChar=='B'){
         SYN_FrameInfo(0, Chinese_gkr); //播放“你好我的朋友”
         Serial.println("green_kill_red");
       }else{
         Serial.println("Error: inputChar invalid");
       }
       delay(4000);
    }
    
}

void SYN_FrameInfo(uint8_t Music, char *HZdata)
{
    char Frame_Info[100];
    unsigned char HZ_Length;
    unsigned char ecc = 0; //定义校验字节
    unsigned int i = 0;

    HZ_Length = strlen(HZdata);        //需要发送文本的长度
    Frame_Info[0] = 0xFD;              //构造帧头FD
    Frame_Info[1] = 0x00;              //构造数据区长度的高字节
    Frame_Info[2] = HZ_Length + 3;     //构造数据区长度的低字节
    Frame_Info[3] = 0x01;              //构造命令字：合成播放命令
    Frame_Info[4] = 0x00 | Music << 4; //构造命令参数：背景音乐设定
    for (i = 0; i < 5; i++)
    {                                //依次发送构造好的5个帧头字节
        ecc = ecc ^ (Frame_Info[i]); //对发送的字节进行异或校验
    }
    for (i = 0; i < HZ_Length; i++)
    {                            //依次发送待合成的文本数据
        ecc = ecc ^ (HZdata[i]); //对发送的字节进行异或校验
    }
    memcpy(&Frame_Info[5], HZdata, HZ_Length);
    Frame_Info[5 + HZ_Length] = ecc;
    for (i = 0; i < 5 + HZ_Length + 1; i++)
    {
        mySerial.write(Frame_Info[i]);
    }
}

