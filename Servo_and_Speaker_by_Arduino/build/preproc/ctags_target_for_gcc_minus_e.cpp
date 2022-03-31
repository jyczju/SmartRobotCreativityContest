# 1 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\mg90s_test.ino"
# 2 "d:\\浙江大学\\科研竞赛\\SmartRobotCreativityContest\\SmartRobotCreativityContest\\Servo_and_Speaker_by_Arduino\\mg90s_test.ino" 2

Servo myservo; // 定义Servo对象来控制
int pos = 0; // 角度存储变量

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
        delay(5); // 等待转动到指定角度
    }
    for (pos = 180; pos >= 0; pos--)
    { // 从180°到0°
        myservo.write(pos); // 舵机角度写入
        delay(5); // 等待转动到指定角度
    }

    // myservo.write(0);
    // delay(1000);
    // myservo.write(180);
    // delay(1000);
}
