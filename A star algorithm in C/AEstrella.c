//MARTA MATUTE DE AMORES - 1496672

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<stdbool.h>

#define R 6371
#define PI 3.141592653589793238

typedef struct{
    char carrer[12];
    int numnode; // fa referència a la posició al vector de nodes
    double llargada;
}infoaresta;

typedef struct node{
    long int id;
    double latitud,longitud;
    int narst;
    infoaresta *arestes;
    bool inCuaOberta, inCuaTancada;
    double f, g, h;
    struct node *nodeanterior;
}node;

typedef struct{
    node *nodes;
    int nnodes;
}mapa;

typedef struct Element {
    node * ndcua;
    struct Element * seg;
}ElementCua;

typedef struct {
    ElementCua *inici;
}UnaCua;

double distancia (node n1, node n2);
int posarencua(UnaCua * cua, node *nounode);
unsigned buscapunt(node llista[], int len, long int ident);
node *treuelprimer(UnaCua *cua);
void treurelement(UnaCua *cua, node *eliminar);
void llegirFitxers(char * fixter1, char *fitxer2, mapa *);

int main(int argc, char const *argv[]) {
    int i, llargadacami = 1;
    node *nodes, *nodeactual, *nodeadjacent, *cami, *aux;
    int posorigen, posfinal;
    long int origen, final;
    UnaCua *cuaoberta, *cuatancada;
    ElementCua *primerElement;
    double novaG;
    mapa mapa;

    /*
    COMPROVACIO D'ERRORS PER A UN CORRECTE FUNCIONAMENT DEL PROGRAMA
    */

    if(argc != 3) {
        fprintf(stderr, "El nombre d'arguements es diferent a 2.\n");
        return 1;
    }
    origen = atol(argv[1]);
    final = atol(argv[2]);

    llegirFitxers("Nodes.csv", "Carrers.csv", &mapa);
    nodes = mapa.nodes;
    posorigen = buscapunt(nodes, mapa.nnodes, origen);
    posfinal = buscapunt(nodes, mapa.nnodes, final);
    if(posorigen == -1){
        printf("El node origen no es troba a la llista de nodes\n");
        return 1;
    }
    if(posfinal == -1){
        printf("El node final no es troba a la llista de nodes\n");
        return 1;
    }

    if(origen == final){
        printf("Els nodes origen i desti son iguals.\n");
        return 1;
    }


    /*
    INICIALITZACIO LLISTES OBERTA I TANCADA
    */

    if((primerElement=(ElementCua*)malloc(sizeof(ElementCua)))==NULL){
        printf("No hi ha prou espai en memoria\n");
        return 1;
    }
    primerElement->ndcua = &nodes[posorigen];
    primerElement->ndcua->g = 0;
    primerElement->ndcua->h = distancia(nodes[posorigen],nodes[posfinal]);
    primerElement->ndcua->f = primerElement->ndcua->g + primerElement->ndcua->h;
    primerElement->seg = NULL;
    primerElement->ndcua->inCuaOberta = 1;
    if((cuaoberta = (UnaCua*)malloc(sizeof(UnaCua)))==NULL
    || (cuatancada = (UnaCua*)malloc(sizeof(UnaCua)))==NULL){
        printf("No hi ha prou espai en memoria\n");
        return 1;
    }
    cuaoberta->inici = primerElement;

    /*
    ALGORITME A*
    */

    while (cuaoberta->inici) {
        nodeactual = treuelprimer(cuaoberta);
        //no cal fer comprovacio com a treurelement o com a posarencua ja que
        //si nodeactual fos NULL aleshores ja no hauriem entrat al while.
        nodeactual->inCuaOberta = false;
        nodeactual->f = nodeactual->h + nodeactual->g;
        if(nodeactual->id == final) {
            break;
        }

        for(i=0; i<nodeactual->narst; i++) {
            nodeadjacent =  &nodes[nodeactual->arestes[i].numnode];
            novaG = nodeactual->g + distancia(*nodeactual, *nodeadjacent);
            if (nodeadjacent->inCuaOberta) {
                if (nodeadjacent->g <= novaG) continue;
            }
            else if (nodeadjacent->inCuaTancada) {
                if (nodeadjacent->g <= novaG) continue;
                treurelement(cuatancada, nodeadjacent);
                nodeadjacent->inCuaTancada = false;
                posarencua(cuaoberta, nodeadjacent);
                nodeadjacent->inCuaOberta = true;
            }
            else {
                nodeadjacent->g = novaG;
                nodeadjacent->h = distancia(*nodeadjacent, nodes[posfinal]);
                posarencua(cuaoberta, nodeadjacent);
                nodeadjacent->inCuaOberta = true;
            }
            nodeadjacent->nodeanterior = nodeactual;
        }
        posarencua(cuatancada, nodeactual);
        nodeactual->inCuaTancada = true;
    }

    /*
    ASSIGNACIO I IMPRESSIO PER PANTALLA DEL CAMI TROBAT
    */

    if (nodeactual->id != nodes[posfinal].id) {
        printf("La llista oberta es buida i no hem trobat un cami\n");
        return 2;
    }

    aux = nodeactual;
    while(nodeactual->id != nodes[posorigen].id){
        llargadacami++;
        nodeactual = nodeactual->nodeanterior;
    }
    nodeactual = aux;
    if((cami = (node *) malloc(llargadacami*sizeof(node)))==NULL) {
        printf("No hi ha prou espai a la memoria.\n");
        return 1;
    }
    for (i=llargadacami-1; i>=0; i--) {
        cami[i] = *nodeactual;
        nodeactual = nodeactual->nodeanterior;
    }

    printf("Cami optim:\n\n");
    for (i=0; i<llargadacami; i++){
        printf(" Id=%010ld | %lf | %lf | Dist=%06.2f m\n",cami[i].id, cami[i].latitud,
        cami[i].longitud, cami[i].g);
    }
    printf("\nLa distancia que ens caldra recorrer es de %lf metres.\n",cami[llargadacami-1].g );
    return 0;
}

void llegirFitxers(char * fitxer1, char *fitxer2, mapa *mapa) {
    FILE *fitxernodes, *fitxercarrers;
    char c, id[10];
    int i, nnodes=0, ncarrers=0, pos1, pos2;
    long int carrer1, carrer2;
    node *nodes;

    if((fitxernodes=fopen(fitxer1,"r"))==NULL) {
        printf("No s'ha accedit al fitxer de dades %s\n", fitxer1);
        exit(1);
    }
    while((c=fgetc(fitxernodes))!=EOF) {
        if(c=='\n') nnodes++;
    }
    if((nodes = (node*) malloc(nnodes*sizeof(node)))==NULL){
        printf("No hi ha prou espai en memoria.\n");
        exit(2);
    }
    rewind(fitxernodes);
    for (i=0; i<nnodes; i++){
        fscanf(fitxernodes,"%ld;%lf;%lf", &nodes[i].id, &nodes[i].latitud,
                                          &nodes[i].longitud);
        nodes[i].inCuaOberta = 0;
        nodes[i].inCuaTancada = 0;
    }
    if((fitxercarrers=fopen(fitxer2,"r"))==NULL) {
        printf("No s'ha accedit al fitxer de dades %s\n", fitxer2);
        exit(3);
    }
    while((c=fgetc(fitxercarrers))!=EOF) {
        if(c=='\n') ncarrers++;
    }
    rewind(fitxercarrers);
    while(fgetc(fitxercarrers) != EOF) {
        fscanf(fitxercarrers, "d=%10s;", id);
        fscanf(fitxercarrers, "%ld", &carrer1);
        while (fgetc(fitxercarrers)!='\n') {
            fscanf(fitxercarrers, "%ld", &carrer2);
            pos1 = buscapunt(nodes, nnodes, carrer1);
            pos2 = buscapunt(nodes, nnodes, carrer2);
            nodes[pos1].narst++;
            if((nodes[pos1].arestes = (infoaresta*) realloc(nodes[pos1].arestes,
                nodes[pos1].narst*sizeof(infoaresta)))==NULL){
                printf("No hi ha prou espai en memoria\n");
                exit(4);
            }
            strcpy(nodes[pos1].arestes[nodes[pos1].narst -1].carrer, id);
            nodes[pos1].arestes[nodes[pos1].narst-1].numnode = pos2;
            nodes[pos1].arestes[nodes[pos1].narst-1].llargada =
                                            distancia(nodes[pos1], nodes[pos2]);
            nodes[pos2].narst++;
            if((nodes[pos2].arestes = (infoaresta*) realloc(nodes[pos2].arestes,
                nodes[pos2].narst*sizeof(infoaresta)))==NULL){
                printf("No hi ha prou espai en memoria\n");
                exit(5);
            }
            strcpy(nodes[pos2].arestes[nodes[pos2].narst-1].carrer, id);
            nodes[pos2].arestes[nodes[pos2].narst-1].numnode = pos1;
            nodes[pos1].arestes[nodes[pos1].narst-1].llargada =
                                            distancia(nodes[pos1], nodes[pos2]);
            carrer1 = carrer2;
        }
    }
    mapa->nodes = nodes;
    mapa->nnodes = nnodes;
}

int posarencua(UnaCua * cua, node *nounode) {
    ElementCua *nou, *aux, *segaux;
    if( (nou = (ElementCua*)malloc(sizeof(ElementCua))) == NULL
     || (aux = (ElementCua*)malloc(sizeof(ElementCua))) == NULL
     || (segaux = (ElementCua*)malloc(sizeof(ElementCua))) == NULL){
        printf("No hi ha prou espai a la memoria\n");
        exit(1);
    }
    nou->ndcua = nounode;
    nounode->f = nounode->g + nounode->h;
    if(!cua->inici) {
        cua->inici = nou;
        cua->inici->seg = NULL;
        return 0;
    }
    if (nounode->f <= (cua->inici->ndcua->g+cua->inici->ndcua->h)) {
        nou->seg = cua->inici;
        cua->inici = nou;
        return 1;
    }
    aux = cua->inici;
    segaux = cua->inici->seg;
    while(segaux) {
        if(nounode->f <= (segaux->ndcua->g+segaux->ndcua->h)){
            aux->seg = nou;
            nou->seg = segaux;
            return 2;
        }
        aux = segaux;
        segaux = aux->seg;
    }
    if(segaux == NULL && nounode->f > (aux->ndcua->g+aux->ndcua->h)){
        aux->seg = nou;
        nou->seg = NULL;
        return 3;
    }
    printf("Hi ha hagut un problema afegint el node %ld a la cua\n",
            nounode->id);
    exit(4);
}

node *treuelprimer(UnaCua *cua){
    node *tret;
    ElementCua *aux;
    if((aux = (ElementCua*)malloc(sizeof(ElementCua))) == NULL
    || (tret = (node*)malloc(sizeof(node)))==NULL) {
        printf("No hi ha suficient espai a la memoria\n");
        exit(1);
    }
    aux = cua->inici;
    cua->inici = cua->inici->seg;
    tret = aux->ndcua;
    free(aux);
    return tret;
}

void treurelement(UnaCua *cua, node *eliminar) {
    ElementCua *aux, *anterior;
    if((aux = (ElementCua*)malloc(sizeof(ElementCua))) == NULL
    || (anterior = (ElementCua*)malloc(sizeof(ElementCua)))==NULL) {
        printf("No hi ha suficient espai a la memoria\n");
        exit(1);
    }
    aux = cua->inici;
    if(aux->ndcua == eliminar){
        aux = cua->inici;
        cua->inici = cua->inici->seg;
        free(aux);
    }
    while(aux->ndcua != eliminar && aux!=NULL) {
        anterior = aux;
        aux = aux->seg;
    }
    if(aux!=NULL && aux->ndcua == eliminar) {
        anterior->seg = aux->seg;
        free(aux);
    }
    else{
        printf("No s'ha trobat el node a eliminar a la cua.\n");
        exit(2);
    }
}

unsigned buscapunt(node llista[], int len, long int ident){
    int a = 0, b=len;
    int c;
    while (a<=b) {
        c=(int)(b+a)/2;
        if (llista[c].id == ident) {
            return c;
        }
        else if (llista[c].id < ident) {
            a=c;
        }
        else {
            b=c;
        }
    }
    //printf("No s'ha trobat l'element %ld a la llista\n", ident);
    return -1;
}

double distancia (node n1, node n2) {
    double x1,y1,z1;
    double x2,y2,z2;
    double lon1, lat1,lon2,lat2;
    lon1=n1.longitud *(PI/180);
    lat1=n1.latitud *(PI/180);
    lon2=n2.longitud*(PI/180);
    lat2=n2.latitud*(PI/180);
    x1=R*cos(lon1)*cos(lat1);
    x2=R*cos(lon2)*cos(lat2);
    y1=R*sin(lon1)*cos(lat1);
    y2=R*sin(lon2)*cos(lat2);
    z1=R*sin(lat1);
    z2=R*sin(lat2);
    double distancia=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
    return distancia*1000;
}
