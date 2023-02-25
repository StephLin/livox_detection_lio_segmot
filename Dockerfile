FROM nvcr.io/nvidia/tensorflow:20.11-tf1-py3

# ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt install -y ros-melodic-ros-base && \
    apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential && \
    rosdep init && \
    rosdep update

RUN apt install -y ros-melodic-jsk-recognition-msgs

RUN python3 -m pip install rospkg catkin_pkg

RUN python3 -m pip install --upgrade scipy

COPY . /livox-detection
WORKDIR /livox-detection

RUN cd utils/lib_cpp && \
    git clone https://github.com/pybind/pybind11.git && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && make && \
    cp lib_cpp.so ../../../

CMD ["bash", "-c", "python3 ros_main.py"]
