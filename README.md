### Рекомендательная система для BootCamp Магнит.

Создаем образ:

```bash
docker build --pull --rm -f "Dockerfile" -t my-project "."
```

Запускаем:

``` bash
docker run --rm -d -p 5000:5000 -v ~/my_project/log:/my_project/log -v ~/my_project/data:/my_project/data --name my_script my-project:latest
```

Алгоритм ожидает входные данные в виде pandas DataFrame c столбцом UID. Выходной DataFrame с столбцами UID и recommendations. 
