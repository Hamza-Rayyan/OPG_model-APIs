version: '3'
services:
    web:
        build: './web'
        ports:
            -"5000:5000"
