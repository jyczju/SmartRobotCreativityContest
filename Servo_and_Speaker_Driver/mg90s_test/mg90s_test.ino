//舵机的棕色线接GND，红色线接5V，黄色线为信号线，接PWM 10


#include <Servo.h>

Servo servo1; // 定义Servo对象来控制
Servo servo2; // 定义Servo对象来控制
int pos = 0;   // 角度存储变量

void Serco_Syn_Write(int angle);

void setup()
{
    servo1.attach(10); // 控制线连接数字10
    servo2.attach(6); // 控制线连接数字10
}

void loop()
{
//    for (pos = 0; pos <= 180; pos++)
//    { // 0°到180°
//        // in steps of 1 degree
//        myservo.write(pos); // 舵机角度写入
//        delay(5);           // 等待转动到指定角度
//    }
//    for (pos = 180; pos >= 0; pos--)
//    {                       // 从180°到0°
//        myservo.write(pos); // 舵机角度写入
//        delay(5);           // 等待转动到指定角度
//    }
//
//    servo1.write(0);
//    servo2.write(0);
    Serco_Syn_Write(0);
    delay(1000);
//    servo1.write(180);
//    servo2.write(180);
    Serco_Syn_Write(180);
    delay(1000);
}

void Serco_Syn_Write(int angle)
{
    servo1.write(angle);
    servo2.write(angle);
}
