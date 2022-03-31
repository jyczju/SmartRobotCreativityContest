//舵机的棕色线接GND，红色线接5V，黄色线为信号线，接PWM 9


#include <Servo.h>

Servo myservo; // 定义Servo对象来控制
int pos = 0;   // 角度存储变量

void setup()
{
    myservo.attach(9); // 控制线连接数字9
}

void loop()
{
    for (pos = 0; pos <= 180; pos++)
    { // 0°到180°
        // in steps of 1 degree
        myservo.write(pos); // 舵机角度写入
        delay(5);           // 等待转动到指定角度
    }
    for (pos = 180; pos >= 0; pos--)
    {                       // 从180°到0°
        myservo.write(pos); // 舵机角度写入
        delay(5);           // 等待转动到指定角度
    }

    // myservo.write(0);
    // delay(1000);
    // myservo.write(180);
    // delay(1000);
}
