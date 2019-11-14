
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need [Docker](https://docs.docker.com/install/), [Docker Compose](https://docs.docker.com/compose/install/) installed. 


### Installing

#### Running all containers

Important note: 
The project generates SSL/TLS certificates and passwords if none are found in the subrepo containing the secrets. If the VIA is going to accessed, insert the real password into ```ikoncode_secrets```.

A step by step series of examples that tell you how to get a development env running

First you need to clone the repository.
In order to do that navigate to the folder where you want to save the project and execute:

```
git clone https://github.com/wittenator/pimai.git
```
Then proceed by building and running the containers:
```
cd pimai/
bash ./start.sh
```

## Authors
* [Tim Korjakow](https://github.com/wittenator)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
