#include<stdio.h>
#include<locale.h>

unsigned char info[54];//Заголовок
unsigned char * imageData;//Данные рисунка
int dataSize;//Кол-во данных
int imageWidth, imageHeight;//Размеры рисунка
int lensX, lensY, radius = 150;//Параметры линзы
float koeff = -0.08;

unsigned char *gpuOrig, *gpuTemp;

__global__ void process(unsigned char *temp, unsigned char *orig, float lensX, float lensY, int radius, int imageHeight, int imageWidth, int size, float koeff)
{
	int x = blockIdx.x*20 + threadIdx.x;
	int y = blockIdx.y*20 + threadIdx.y;
	if(y * imageWidth * 3 + x * 3 >= size) {
		return;
	} 

	float toLens = sqrt((lensX-x)*(lensX-x) + (lensY-y)*(lensY-y));
	if(toLens<radius) {
		int xn = lensX - x, yn = lensY - y;
		float dispose = 1 + toLens*koeff/radius;
		//Получаем координаты с учетом коэффициента линзы
        int x_old = lensX - round(xn*dispose), y_old = lensY - round(yn*dispose);

		if(x_old < 0) {
			x_old = 0;
		} else if(x_old >= imageWidth) {
			x_old = imageWidth-1;
		}

		if(y_old < 0) {
			y_old = 0;
		} else if(y_old >= imageHeight) {
			y_old = imageHeight-1;
		}

		temp[y * imageWidth * 3 + x * 3] = orig[y_old * imageWidth * 3 + x_old * 3];
		temp[y * imageWidth * 3 + x * 3 + 1] = orig[y_old * imageWidth * 3 + x_old * 3+1];
		temp[y * imageWidth * 3 + x * 3 + 2] = orig[y_old * imageWidth * 3 + x_old * 3+2];
	}
}

__host__ int main() {
	setlocale(LC_ALL, ".ACP");

	printf("Начинаю открытия изображения \"image.bmp\"\n");

	clock_t start;
	start = clock();

	//====================
	//Открытие изображения
	//====================
	FILE* f = fopen("image.bmp", "rb");//Открываем
	fread(info, sizeof(unsigned char), 54, f); //Считываем 54 битный заголовок

	//Получаем информацию о размере из заголовка
    imageWidth = *(int*)&info[18];
    imageHeight = *(int*)&info[22];

	//Получаем количество байтов, включая нулевые
    int size = size = imageWidth * 3 + imageWidth % 4;
    size = size * imageHeight;
    dataSize = size;

	//Задаем значения для чтения данных пикселей
    int k = 0, widthSize = imageWidth * 3 + imageWidth % 4, line = 1;;
    imageData = new unsigned char[size];

	unsigned char readbyte;//Читаемое значение
    for(int i = 0;i<size; i++) {//Проходим по всем данным
        fread(&readbyte, sizeof(unsigned char), 1, f);//считываем байт

        //Пропуск нулевых байтов
        if(i == widthSize * line) line++;
        if(widthSize * line - i <= imageWidth % 4) {
            continue;
        }

        //Запись данных
        imageData[k++] = readbyte;
    }
    fclose(f);//закрываем
	//переводим из формата BGR в RGB
    for(int i = 0; i < size; i += 3) {
            unsigned char tmp = imageData[i];
            imageData[i] = imageData[i+2];
            imageData[i+2] = tmp;
    }
	printf("Время открытия изображения %d мс\n\n", (int)(clock()-start));
	//Задаем параметры линзы
    lensX = imageWidth / 2;
    lensY = imageHeight / 2;
	koeff = -0.3;
	printf("Выделяю память CUDA\n");
	start = clock();
	//Выделяем память под изображение(оригинал и которое будет изменяться) на видеокарте
	cudaMalloc((void**)&gpuOrig, size);
	cudaMalloc((void**)&gpuTemp, size);
	cudaMemcpy(gpuOrig, imageData, size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuTemp, imageData, size, cudaMemcpyHostToDevice);
	printf("Время выделения памяти %d мс\n\n", (int)(clock()-start));
	printf("Начинаю обрабатывать изображение\n");
	start = clock();

	//Задаем параметры grid'а и block'ов
	dim3 gridSize = dim3(imageWidth/20+1, imageHeight/20+1, 1);   
	dim3 blockSize = dim3(20, 20, 1);
	process<<<gridSize, blockSize>>>(gpuTemp, gpuOrig, lensX, lensY, radius, imageHeight, imageWidth, size, koeff);

	//Синхронизируем(для того чтобы убедится что все потоки закончили работу)
	cudaEvent_t syncEvent;
	cudaEventCreate(&syncEvent);    //Создаем event
	cudaEventRecord(syncEvent, 0);  //Записываем event
	cudaEventSynchronize(syncEvent);  //Синхронизируем event

	printf("Время обработки изображения %d мс\n\n", (int)(clock()-start));

	printf("Начинаю выгружать обработанное изображение из GPU\n");
	start = clock();

	//Копируем изображения из памяти видеокарты в оперативную
	cudaMemcpy(imageData, gpuTemp, size, cudaMemcpyDeviceToHost);

	printf("Время выгрузки изображения %d мс\n\n", (int)(clock()-start));

	printf("Начинаю сохранение обработанного изображения(new.bmp)\n");
	start = clock();
	//======================
	//Сохранение изображения
	//======================
	f = fopen("new.bmp", "wb");//Открываем файл
    fwrite(info, sizeof(unsigned char), 54, f); //Сохраняем заголовок
    size = imageWidth * 3 + imageWidth % 4;
    size = size * imageHeight;//Определяем кол-во байтов с нулевыми

    unsigned char nullbuff[3];//нулевой буффер
    for(int i = 0; i < 3; i++) {
		nullbuff[i] = 0;
    }
    //Производим считывание обработанного рисунка
	int savedWidth = 0;//Перменная для хранения текущего кол-ва сохраненных байтов(для добавления нулей в конец)
	unsigned char bybuff[3];//данные пикселя(формат BGR)
	for(int i = 0;i<size; i++) {//Проходим по всем данным
		//Задаем данные пикселя
		bybuff[0] = imageData[i+2];
        bybuff[1] = imageData[i+1];
        bybuff[2] = imageData[i];
		i+=2;//пропускаем 2 следующих бита(мы их уже считали)
	    savedWidth++;//увеличиваем ширину считанных пикселей
		fwrite(bybuff, sizeof(unsigned char), 3, f);//записываем данные пикселя

		//Если считав строку требуются нулевые байты - добавялем
        if(imageWidth % 4 != 0 && savedWidth == imageWidth) {
			savedWidth = 0;
            fwrite(nullbuff, sizeof(unsigned char), imageWidth % 4, f);
        }
	}
    fclose(f);//Закрываем
	printf("Время сохранения изображения %d мс\n\n", (int)(clock()-start));
	//Освобождаем память
	cudaFree(gpuTemp);
	cudaFree(gpuOrig);
	delete []imageData;
	return 0;
}
