echo "Building"
./gradlew build
./gradlew shadowJar
echo "Clearing remote"
ssh timeweb "cd hotel; rm -rf build/ .env.sample Dockerfile docker-compose.yml sql/"
echo "Syncing build dir"
ssh timeweb "cd hotel; mkdir build; mkdir build/libs"
scp -r build/libs timeweb:hotel/build
echo "Syncing env"
scp .env.sample timeweb:hotel
echo "Syncing Dockerfile"
scp Dockerfile timeweb:hotel
echo "Syncing docker-compose"
scp docker-compose.yml timeweb:hotel
echo "Syncing sql"
scp -r sql timeweb:hotel
echo "Starting server"
ssh timeweb "cd hotel; docker-compose down && docker-compose up --build -d"
echo "DONE"