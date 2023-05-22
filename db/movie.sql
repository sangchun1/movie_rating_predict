CREATE DATABASE movie;

create user 'project'@'%' identified by '1234';
create user 'project'@'localhost' identified by '1234';
flush PRIVILEGES;

grant all privileges on movie.* to 'project'@'%';
grant all privileges on movie.* to 'project'@'localhost';