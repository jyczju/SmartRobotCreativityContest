#include <Arduino.h>
#line 1 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_Driver\\speaker\\speaker.ino"
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
char Chinese_welcome[40] = {char(0xBB), char(0xB6), char(0xD3), char(0xAD), char(0xCA), char(0xB9), char(0xD3), char(0xC3), char(0xD6), char(0xC7), char(0xC4), char(0xDC), char(0xBE), char(0xFC), char(0xC6), char(0xEC), char(0xD6), char(0xFA), char(0xCA), char(0xD6)}; //欢迎使用智能军旗助手
char Chinese_gw[20] = {char(0xC2), char(0xCC), char(0xB7), char(0xBD), char(0xBB), char(0xF1), char(0xCA), char(0xA4)}; //绿方获胜
char Chinese_rw[20] = {char(0xBA), char(0xEC), char(0xB7), char(0xBD), char(0xBB), char(0xF1), char(0xCA), char(0xA4)}; //红方获胜
char Chinese_es[20] = {char(0xCB), char(0xAB), char(0xB7), char(0xBD), char(0xCB), char(0xBE), char(0xC1), char(0xEE), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //双方司令阵亡
char Chinese_rskbz[20] = {char(0xBA), char(0xEC), char(0xB7), char(0xBD), char(0xCB), char(0xBE), char(0xC1), char(0xEE), char(0xB1), char(0xBB), char(0xD5), char(0xA8), char(0xCB), char(0xC0)}; //红方司令被炸死
char Chinese_gskbz[20] = {char(0xC2), char(0xCC), char(0xB7), char(0xBD), char(0xCB), char(0xBE), char(0xC1), char(0xEE), char(0xB1), char(0xBB), char(0xD5), char(0xA8), char(0xCB), char(0xC0)}; //绿方司令被炸死
char Chinese_equal[20] = {char(0xCB), char(0xAB), char(0xB7), char(0xBD), char(0xC6), char(0xE5), char(0xD7), char(0xD3), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //双方棋子阵亡
char Chinese_rskbd[20] = {char(0xBA), char(0xEC), char(0xB7), char(0xBD), char(0xCB), char(0xBE), char(0xC1), char(0xEE), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //红方司令阵亡
char Chinese_gskbd[20] = {char(0xC2), char(0xCC), char(0xB7), char(0xBD), char(0xCB), char(0xBE), char(0xC1), char(0xEE), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //绿方司令阵亡
char Chinese_rkg[20] = {char(0xC2), char(0xCC), char(0xB7), char(0xBD), char(0xC6), char(0xE5), char(0xD7), char(0xD3), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //绿方棋子阵亡
char Chinese_gkr[20] = {char(0xBA), char(0xEC), char(0xB7), char(0xBD), char(0xC6), char(0xE5), char(0xD7), char(0xD3), char(0xD5), char(0xF3), char(0xCD), char(0xF6)}; //红方棋子阵亡

char inputChar = 'Z';
int DELAY_TIME = 3500;

void SYN_FrameInfo(uint8_t Music, char *HZdata);

#line 30 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_Driver\\speaker\\speaker.ino"
void setup();
#line 43 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_Driver\\speaker\\speaker.ino"
void loop();
#line 30 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_Driver\\speaker\\speaker.ino"
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
    delay(DELAY_TIME);
}

void loop()
{
    //串口数据读取
    if(Serial.available())
    {
       inputChar = Serial.read(); //char(Serial.read())
       Serial.println(inputChar);
       if (inputChar=='A'){
         SYN_FrameInfo(0, Chinese_gw); //播放“绿方获胜” 
         Serial.println("green_win");
         delay(DELAY_TIME);
       }else if(inputChar=='B'){
         SYN_FrameInfo(0, Chinese_rw); //播放“红方获胜”
         Serial.println("red_win");
         delay(DELAY_TIME);
       }else if(inputChar=='C'){
         SYN_FrameInfo(0, Chinese_es); //播放“双方司令阵亡”
         Serial.println("equal_siling");
         delay(DELAY_TIME);
       }else if(inputChar=='D'){
         SYN_FrameInfo(0, Chinese_rskbz); //播放“红方司令被炸死”
         Serial.println("red_siling_killed_by_zhadan");
         delay(DELAY_TIME);
       }else if(inputChar=='E'){
         SYN_FrameInfo(0, Chinese_gskbz); //播放“绿方司令被炸死”
         Serial.println("green_siling_killed_by_zhadan");
         delay(DELAY_TIME);
       }else if(inputChar=='F'){
         SYN_FrameInfo(0, Chinese_equal); //播放“双方棋子阵亡”
         Serial.println("equal");
         delay(DELAY_TIME);
       }else if(inputChar=='G'){
         SYN_FrameInfo(0, Chinese_rskbd); //播放“红方司令阵亡”
         Serial.println("red_siling_killed_by_dilei");
         delay(DELAY_TIME);
       }else if(inputChar=='H'){
         SYN_FrameInfo(0, Chinese_gskbd); //播放“绿方司令阵亡”
         Serial.println("green_siling_killed_by_dilei");
         delay(DELAY_TIME);
       }else if(inputChar=='I'){
         SYN_FrameInfo(0, Chinese_rkg); //播放“绿方棋子阵亡”
         Serial.println("red_kill_green");
         delay(DELAY_TIME);
       }else if(inputChar=='J'){
         SYN_FrameInfo(0, Chinese_gkr); //播放“红方棋子阵亡”
         Serial.println("green_kill_red");
         delay(DELAY_TIME);
       }else if(inputChar=='K'){
         SYN_FrameInfo(0, Chinese_rkg); //播放“绿方棋子阵亡”
         Serial.println("red_kill_green");
         delay(DELAY_TIME);
       }else if(inputChar=='L'){
         SYN_FrameInfo(0, Chinese_gkr); //播放“红方棋子阵亡”
         Serial.println("green_kill_red");
         delay(DELAY_TIME);
       }else{
         Serial.println("Error: inputChar invalid");
       }
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

