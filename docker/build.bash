docker build \
    -f docker/Dockerfile \
    --build-arg MY_USER_ID=$(id -u) \
    -t docsaid_training_base_image .

