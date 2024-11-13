FROM gradle:jdk-21-and-22-graal

RUN git clone --depth 1 https://github.com/breandan/cstk.git && \
    cd cstk && \
    git submodule update --init --recursive --remote

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN cd cstk && ./gradlew shadowJar

CMD cd cstk && ./gradlew pythonBarHillel --stacktrace

# To run the image, use the following command:
# docker run -it --rm breandan/tacas25

# To build the image, use the following command:
# docker build -t breandan/tacas25 .

# To push the image to Docker Hub, use the following command:
# docker push breandan/tacas25