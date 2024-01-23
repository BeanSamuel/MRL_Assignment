#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "fstream"
#include "chrono"
#include "iostream"
#include "thread"
#define RAD2DEF(x) ((x)*180./M_PI)

std::ofstream data("./src/hw1/src/data.dat",std::ofstream::out);

auto start=std::chrono::system_clock::now();

auto end=std::chrono::system_clock::now();

void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan){
  end=std::chrono::system_clock::now();
  std::chrono::duration<double> el_time=end-start;
  if(el_time.count()>=60.0)exit(0);
  for(int i=0;i<360;i++){
   
    data<<scan->ranges[i]<<" "<<RAD2DEF(scan->angle_min+scan->angle_increment*i)<<" "<<el_time.count()<<std::endl;
  }
  std::cout<<el_time.count()<<std::endl;
}

int main(int argc, char **argv){
  
  ros::init(argc, argv,"listener");

  ros::NodeHandle n;

  ros::Subscriber sub4 = n.subscribe<sensor_msgs::LaserScan>("/scan",500,scanCallback);

  ros::spin();
  return 0;
}