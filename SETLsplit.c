#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

/***********************************************************
  Simple circular linked list for match records
 ***********************************************************/

typedef struct MSTRUCT {
	int iteration, row, col, rotation;
	struct MSTRUCT *next;
} MATCH;


typedef struct {
	int nItem;
	MATCH* tail;
	MATCH* head;
} MATCHLIST;

MATCHLIST* newList();

void deleteList( MATCHLIST*);

void insertEnd(MATCHLIST*, int, int, int, int);

void printList(MATCHLIST*);

/***********************************************************
  Helper functions 
 ***********************************************************/
//to collect evolved world from sub processes
void gatherEvo(char **nextW, int tag, int numSlaveProcess, int wSize);

// To distribute world for evolution
void distributeEvo(char **currW, int numSlaveProcess, int tag, int wSize);

// To distribute matrix size to all processes
int distributeSize(int size, int numSlaveProcess, int rank);

// To distribute pattern to all processes
int distributePattern(char ***, int patternSize, int numSlaveProcess, int rank);

// For all other slave p to receive work
void receiveWork(char **recvBuf, int numRows, int wSize, int tag);

// For p0 to distribute data
void distributeWork(char **, int numSlaveProcess, int tag, int wSize, int pSize);

// To Init parallelization
void init(int* argc, char** argv, int* numTask, int* rank);

//For exiting on error condition
void die(int lineNo);

//For trackinng execution
long long wallClockTime();


/***********************************************************
  Square matrix related functions, used by both world and pattern
 ***********************************************************/
// Receives list and appends to main list
void gatherWork(int *, int tag, int numTask, MATCHLIST* list);

// Convert searchPatterns result into array
int* convertMatchListToArr(MATCHLIST* list, int rowOffset);

// For generation of buffer usage
char ** allocateEmptySquareMatrix(int row, int size);

char** allocateSquareMatrix( int size, char defaultValue );

void freeSquareMatrix( char** );

void printSquareMatrix( char**, int size );


/***********************************************************
  World  related functions
 ***********************************************************/

#define ALIVE 'X' 
#define DEAD 'O'

char** readWorldFromFile( char* fname, int* size );

int countNeighbours(char** world, int row, int col);

void evolveWorld(char** curWorld, char** nextWorld, int row, int size);

/***********************************************************
  Search related functions
 ***********************************************************/

//Using the compass direction to indicate the rotation of pattern
#define N 0 //no rotation
#define E 1 //90 degree clockwise
#define S 2 //180 degree clockwise
#define W 3 //90 degree anti-clockwise

char** readPatternFromFile( char* fname, int* size );

void rotate90(char** current, char** rotated, int size);

void searchPatterns(char** world, int wSize, int iteration, 
		char** patterns[4], int pSize, MATCHLIST* list, int row);

void searchSinglePattern(char** world, int wSize, int interation,
		char** pattern, int pSize, int rotation, MATCHLIST* list, int row);

/***********************************************************
  Main function
 ***********************************************************/


int main( int argc, char** argv)
{
	char **curW, **nextW, **temp, **recvBuf;
	//char dummy[20];
	char **patterns[4];
	int dir, iterations, iter, rank, numTask, numSlaveProcess, row, tag = 6, rowOffset;
	int size, patternSize;
	long long before, after, beforePrint;
	MATCHLIST*list;
	int *listArr;
	int *resultBuf;
	MPI_Request iSendReq;

	if (argc < 4 ){
		fprintf(stderr, 
				"Usage: %s <world file> <Iterations> <pattern file>\n", argv[0]);
		exit(1);
	} 

	iterations = atoi(argv[2]);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numTask);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	numSlaveProcess = numTask - 1;	

	if (rank == 0){
		curW = readWorldFromFile(argv[1], &size);
		nextW = allocateSquareMatrix(size+2, DEAD);


		printf("World Size = %d\n", size);

		printf("Iterations = %d\n", iterations);

		patterns[N] = readPatternFromFile(argv[3], &patternSize);
		for (dir = E; dir <= W; dir++){
			patterns[dir] = allocateSquareMatrix(patternSize, DEAD);
			rotate90(patterns[dir-1], patterns[dir], patternSize);
		}
		printf("Pattern size = %d\n", patternSize);


		//Start timer
		before = wallClockTime();

	}

#ifdef DEBUG
	printSquareMatrix(patterns[N], patternSize);
	printSquareMatrix(patterns[E], patternSize);
	printSquareMatrix(patterns[S], patternSize);
	printSquareMatrix(patterns[W], patternSize);
#endif
	size = distributeSize(size, numSlaveProcess, rank);
	patternSize = distributePattern(patterns, patternSize, numSlaveProcess, rank);
	
	list = newList();

	for (iter = 0; iter < iterations; iter++){

#ifdef DEBUG
		printf("World Iteration.%d\n", iter);
		printSquareMatrix(curW, size+2);
#endif
		if (rank == 0){
			
			distributeWork(curW, numSlaveProcess, tag, size, patternSize);
			gatherWork(resultBuf, tag, numTask, list);
			//Generate next generation
			tag++;
			// distributeEvo(curW, numSlaveProcess, tag, size);
			gatherEvo(nextW, tag, numSlaveProcess, size);

			temp = curW;
			curW = nextW;
			nextW = temp;
		}
		else if (rank > 0 && rank < numSlaveProcess){
			//recv
			row = floor(size / (float)numSlaveProcess) + patternSize + 1;
			recvBuf = allocateEmptySquareMatrix(row, (size + 2));
			receiveWork(recvBuf, row, size, tag);
			list = newList();

			searchPatterns( recvBuf, size, iter, patterns, patternSize, list, row);
			
			//Send back
			rowOffset = (rank-1) * floor(size / (float)numSlaveProcess);
			listArr = convertMatchListToArr(list, rowOffset);
			MPI_Isend(&(listArr[0]), list->nItem *4, MPI_INT, 0, tag, MPI_COMM_WORLD, &iSendReq);
			if (list != NULL){
				deleteList(list);
			}
			if (listArr != NULL || sizeof(listArr) == 0){
				free(listArr);
			}
			// free(recvBuf);
			tag++;

			//Evolve world
			row = floor(size/ (float)numSlaveProcess) + 2;
			// curW = allocateEmptySquareMatrix(row, size + 2);
			nextW = allocateEmptySquareMatrix(row, size + 2);
			// receiveWork(curW, row, size, tag);
			evolveWorld(recvBuf, nextW, row - 2, size);
			MPI_Send(&(nextW[1][0]), (row-2) * (size + 2), MPI_CHAR, 0, tag, MPI_COMM_WORLD);
			free(recvBuf);
		}
		else if (rank == numSlaveProcess){
			//Last process handling is special due to odd sizes
			if (size%numSlaveProcess == 0){
				row = size / numSlaveProcess + 2; 
			}
			else{
				row = size - (numSlaveProcess - 1) * floor(size/(float)numSlaveProcess) + 2;
			}
			recvBuf = allocateEmptySquareMatrix(row, (size + 2));
			receiveWork(recvBuf, row, size, tag);
			list = newList();
			searchPatterns( recvBuf, size, iter, patterns, patternSize, list, row);
			//Send back
			rowOffset = (rank-1) * floor(size / (float)numSlaveProcess);
			listArr = convertMatchListToArr(list, rowOffset);
			MPI_Isend(&(listArr[0]), list->nItem *4, MPI_INT, 0, tag, MPI_COMM_WORLD, &iSendReq);
			if (list != NULL){
				deleteList(list);
			}
			if (listArr != NULL || sizeof(listArr) == 0){
				free(listArr);
			}
			// free(recvBuf);
			tag++;
			
			//Evolve world	
			if (size%numSlaveProcess == 0){
				row = size / numSlaveProcess + 2; 
			}
			else{
				row = size - (int)((numSlaveProcess - 1) * (floor(size/(float)numSlaveProcess))) + 2;
			}
			// curW = allocateEmptySquareMatrix(row, size + 2);
			nextW = allocateEmptySquareMatrix(row, size + 2);
			// receiveWork(curW, row, size, tag);
			evolveWorld(recvBuf, nextW, row - 2, size);
			MPI_Send(&(nextW[1][0]), (row-2) * (size + 2), MPI_CHAR, 0, tag, MPI_COMM_WORLD);
			free(recvBuf);
		}
		tag++;
	}

	freeSquareMatrix( patterns[0] );
	freeSquareMatrix( patterns[1] );
	freeSquareMatrix( patterns[2] );
	freeSquareMatrix( patterns[3] );

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();	

	if (rank == 0){		
		freeSquareMatrix( curW );
		freeSquareMatrix( nextW );
		beforePrint = wallClockTime();
		printList( list );
		after = wallClockTime();
		deleteList( list );

		//printf("Printing List took %1.2f seconds\n", ((float)(after-beforePrint))/1000000000);
		printf("Parallel SETL took %1.2f seconds\n", 
				((float)(after - before))/1000000000);
	}

	return 0;

}

/***********************************************************
  Helper functions 
 ***********************************************************/
void gatherEvo(char **nextW, int tag, int numSlaveProcess, int wSize){	
	int i, numRows = floor(wSize / (float)numSlaveProcess), rowNum = 1;
	int rowOffset;
	MPI_Status status;	
	MPI_Request req;

	for (i = 1; i < numSlaveProcess; i++){
		MPI_Irecv(&(nextW[rowNum + (i-1) * numRows][0]), numRows * (wSize + 2), MPI_CHAR, i, tag, MPI_COMM_WORLD, &req);
		// rowNum += numRows;
	}

	
	if (wSize%numSlaveProcess == 0){
		rowOffset = wSize / numSlaveProcess; 
	}
	else{
		rowOffset = wSize - (numSlaveProcess - 1) * floor(wSize/(float)numSlaveProcess);
	}

	MPI_Recv(&(nextW[rowNum + (numSlaveProcess - 2) * numRows][0]), rowOffset * (wSize + 2), MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
}



void distributeEvo(char **currW, int numSlaveProcess, int tag, int wSize){	
	int i, numRows = floor(wSize / (float)numSlaveProcess) + 2, rowNum = 0;

	// Distribute to processes 1 to n-1
	for (i = 1; i < numSlaveProcess; i++){
		MPI_Send(&(currW[rowNum][0]), numRows * (wSize + 2), MPI_CHAR, i, tag, MPI_COMM_WORLD);
		rowNum += floor(wSize/(float)numSlaveProcess);
	}

	// Handle last case of odd number of rows seperately
	if (wSize %numSlaveProcess == 0){
		numRows = wSize/numSlaveProcess + 2;
	}
	else{
		numRows = wSize - ((numSlaveProcess - 1) * (floor(wSize/(float)numSlaveProcess))) + 2 ;
	}
	MPI_Send(&(currW[rowNum][0]), numRows * (wSize + 2), MPI_CHAR, numSlaveProcess, tag, MPI_COMM_WORLD);
}

//Distribute size from rank 0 to others
int distributeSize(int size, int numSlaveProcess, int rank){
	int i;
	int recvSize = size;

	if (rank == 0){
		for (i = 1; i<= numSlaveProcess; i++){
			MPI_Send(&size, 1, MPI_INT, i, 5, MPI_COMM_WORLD);
		}	
	}
	else{
		MPI_Status status;
		MPI_Recv(&recvSize, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
	}

	return recvSize;
}

//Distribute pattern from rank 0 to others
int distributePattern(char ***pattern, int patternSize, int numSlaveProcess, int rank){
	int i;
	if (rank == 0){
		for (i = 1; i <= numSlaveProcess; i++){
			MPI_Send(&patternSize, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&(pattern[0][0][0]), patternSize * patternSize, MPI_CHAR, i, 1, MPI_COMM_WORLD);
			MPI_Send(&(pattern[1][0][0]), patternSize * patternSize, MPI_CHAR, i, 2, MPI_COMM_WORLD);
			MPI_Send(&(pattern[2][0][0]), patternSize * patternSize, MPI_CHAR, i, 3, MPI_COMM_WORLD);	
			MPI_Send(&(pattern[3][0][0]), patternSize * patternSize, MPI_CHAR, i, 4, MPI_COMM_WORLD);
		}	
	}
	else{
		MPI_Status status;
		MPI_Recv(&patternSize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		for (i = 0; i < 4; i++){
			pattern[i] = allocateSquareMatrix(patternSize, DEAD);
		}
		MPI_Recv(&(pattern[0][0][0]), patternSize * patternSize, MPI_CHAR, 0, 1, MPI_COMM_WORLD, &status);		
		MPI_Recv(&(pattern[1][0][0]), patternSize * patternSize, MPI_CHAR, 0, 2, MPI_COMM_WORLD, &status);
		MPI_Recv(&(pattern[2][0][0]), patternSize * patternSize, MPI_CHAR, 0, 3, MPI_COMM_WORLD, &status);	
		MPI_Recv(&(pattern[3][0][0]), patternSize * patternSize, MPI_CHAR, 0, 4, MPI_COMM_WORLD, &status);
	}

	return patternSize;
}

// Receives list and appends to main list
void gatherWork(int *recvBuf, int tag, int numTask, MATCHLIST* list){
	int i, j, recvSize, arrIndex;
	MPI_Status status;
	MATCHLIST *lists[4];
	MATCH *temp;

	for (i=0; i < 4; i++){
		lists[i] = newList();
	}	

	//Every process doesn't know how many results they are going to receive
	for (i = 1; i < numTask; i++){
		MPI_Probe(i, tag, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &recvSize);
		if (recvSize > 0){
			recvBuf = (int *)malloc(sizeof(int) * recvSize);
			MPI_Recv(&(recvBuf[0]), recvSize, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

			for (j = 0; j < recvSize ; j += 4){
				arrIndex= recvBuf[j+3];
				insertEnd(lists[arrIndex], recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
			}
		}
	}

	// Counting sort the lists by iterations
	for (i = 0; i < 4; i++){
		if (lists[i]->nItem > 0){
			if (list->nItem > 0){
				temp = list->tail->next;
				list->tail->next = lists[i]->tail->next;
				lists[i]->tail->next = temp;
				list->tail = lists[i]->tail;
			}
			else{
				list->tail = lists[i]->tail;
			}

		}

		list->nItem += lists[i]->nItem;
	}

}

int* convertMatchListToArr(MATCHLIST* list, int rowOffset){
	int i;
	int *arr;
	MATCH* curr;
	if (list->nItem > 0){
		curr = list->tail->next;
	}
	arr = malloc(list->nItem * 4 * sizeof(int));
	for ( i= 0 ; i < list->nItem; i++){
		(arr[i*4]) = curr->iteration;
		(arr[i*4+1]) = curr-> row + rowOffset;
		(arr[i*4+2]) = curr-> col;
		(arr[i*4+3]) = curr-> rotation;
		curr = curr->next;
	}
	return arr;
}

void receiveWork(char **recvBuf, int numRows, int wSize, int tag){
	MPI_Status status;

	MPI_Recv(&(recvBuf[0][0]), numRows * (wSize +2), MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
}

void distributeWork(char ** currW, int numSlaveProcess, int tag, int wSize, int pSize){
	int i, numRows = floor(wSize / (float)numSlaveProcess) + pSize + 1, rowNum = 0;
	int rowOffset;
	MPI_Request req;

	// Distribute to processes 1 to n-1
	for (i = 1; i < numSlaveProcess; i++){
		MPI_Isend(&(currW[rowNum + (i-1) * numRows][0]), numRows * (wSize + 2), MPI_CHAR, i, tag, MPI_COMM_WORLD, &req);
		// rowNum += floor(wSize/(float)numSlaveProcess);
	}

	// Handle last case of odd number of rows seperately
	if (wSize %numSlaveProcess == 0){
		rowOffset = wSize/numSlaveProcess + 2;
	}
	else{
		rowOffset = wSize - ((numSlaveProcess - 1) * (floor(wSize/(float)numSlaveProcess))) + 2 ;
	}
	MPI_Send(&(currW[rowNum + (numSlaveProcess -2) * numRows][0]), rowOffset * (wSize + 2), MPI_CHAR, numSlaveProcess, tag, MPI_COMM_WORLD);
	tag++;
}

void die(int lineNo)
{
	fprintf(stderr, "Error at line %d. Exiting\n", lineNo);
	exit(1);
}

long long wallClockTime( )
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/***********************************************************
  Square matrix related functions, used by both world and pattern
 ***********************************************************/
char ** allocateEmptySquareMatrix(int rows,int size){

	char* contiguous;
	char** matrix;
	int i;

	//Using a least compiler version dependent approach here
	//C99, C11 have a nicer syntax.    
	contiguous = (char*) malloc(sizeof(char) * rows * size);
	if (contiguous == NULL) 
		die(__LINE__);

	//Point the row array to the right place
	matrix = (char**) malloc(sizeof(char*) * size );
	if (matrix == NULL) 
		die(__LINE__);

	matrix[0] = contiguous;
	for (i = 1; i < rows; i++){
		matrix[i] = &contiguous[i*size];
	}

	return matrix;
}

char** allocateSquareMatrix( int size, char defaultValue )
{

	char* contiguous;
	char** matrix;
	int i;

	//Using a least compiler version dependent approach here
	//C99, C11 have a nicer syntax.    
	contiguous = (char*) malloc(sizeof(char) * size * size);
	if (contiguous == NULL) 
		die(__LINE__);


	memset(contiguous, defaultValue, size * size );

	//Point the row array to the right place
	matrix = (char**) malloc(sizeof(char*) * size );
	if (matrix == NULL) 
		die(__LINE__);

	matrix[0] = contiguous;
	for (i = 1; i < size; i++){
		matrix[i] = &contiguous[i*size];
	}

	return matrix;
}

void printSquareMatrix( char** matrix, int size )
{
	int i,j;

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			printf("%c", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void freeSquareMatrix( char** matrix )
{
	if (matrix == NULL) return;

	free( matrix[0] );
}

/***********************************************************
  World  related functions
 ***********************************************************/

char** readWorldFromFile( char* fname, int* sizePtr )
{
	FILE* inf;

	char temp, **world;
	int i, j;
	int size;

	inf = fopen(fname,"r");
	if (inf == NULL)
		die(__LINE__);


	fscanf(inf, "%d", &size);
	fscanf(inf, "%c", &temp);

	//Using the "halo" approach
	// allocated additional top + bottom rows
	// and leftmost and rightmost rows to form a boundary
	// to simplify computation of cell along edges
	world = allocateSquareMatrix( size + 2, DEAD );

	for (i = 1; i <= size; i++){
		for (j = 1; j <= size; j++){
			fscanf(inf, "%c", &world[i][j]);
		}
		fscanf(inf, "%c", &temp);
	}

	*sizePtr = size;    //return size
	return world;

}

int countNeighbours(char** world, int row, int col)
	//Assume 1 <= row, col <= size, no check 
{
	int i, j, count;

	count = 0;
	for(i = row-1; i <= row+1; i++){
		for(j = col-1; j <= col+1; j++){
			count += (world[i][j] == ALIVE );
		}
	}

	//discount the center
	count -= (world[row][col] == ALIVE);

	return count;

}

void evolveWorld(char** curWorld, char** nextWorld, int rowSize, int colSize)
{
	int i, j, liveNeighbours;

	for (i = 1; i <= rowSize; i++){
		for (j = 1; j <= colSize; j++){
			liveNeighbours = countNeighbours(curWorld, i, j);
			nextWorld[i][j] = DEAD;

			//Only take care of alive cases
			if (curWorld[i][j] == ALIVE) {

				if (liveNeighbours == 2 || liveNeighbours == 3)
					nextWorld[i][j] = ALIVE;

			} else if (liveNeighbours == 3)
				nextWorld[i][j] = ALIVE;
		} 
	}
}

/***********************************************************
  Search related functions
 ***********************************************************/

char** readPatternFromFile( char* fname, int* sizePtr )
{
	FILE* inf;

	char temp, **pattern;
	int i, j;
	int size;

	inf = fopen(fname,"r");
	if (inf == NULL)
		die(__LINE__);


	fscanf(inf, "%d", &size);
	fscanf(inf, "%c", &temp);

	pattern = allocateSquareMatrix( size, DEAD );

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			fscanf(inf, "%c", &pattern[i][j]);
		}
		fscanf(inf, "%c", &temp);
	}

	*sizePtr = size;    //return size
	return pattern;
}


void rotate90(char** current, char** rotated, int size)
{
	int i, j;

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			rotated[j][size-i-1] = current[i][j];
		}
	}
}

void searchPatterns(char** world, int wSize, int iteration, 
		char** patterns[4], int pSize, MATCHLIST* list, int row)
{
	int dir;

	for (dir = N; dir <= W; dir++){
		searchSinglePattern(world, wSize, iteration, 
				patterns[dir], pSize, dir, list, row);
	}

}

void searchSinglePattern(char** world, int wSize, int iteration,
		char** pattern, int pSize, int rotation, MATCHLIST* list, int row)
{
	int wRow, wCol, pRow, pCol, match;

	for (wRow = 1; wRow < (row-pSize); wRow++){
		for (wCol = 1; wCol <= (wSize-pSize+1); wCol++){
			match = 1;
#ifdef DEBUGMORE
			printf("S:(%d, %d)\n", wRow-1, wCol-1);
#endif
			for (pRow = 0; match && pRow < pSize; pRow++){
				for (pCol = 0; match && pCol < pSize; pCol++){
					if(world[wRow+pRow][wCol+pCol] != pattern[pRow][pCol]){
#ifdef DEBUGMORE
						printf("\tF:(%d, %d) %c != %c\n", pRow, pCol,
								world[wRow+pRow][wCol+pCol], pattern[pRow][pCol]);
#endif
						match = 0;    
					}
				}
			}
			if (match){
				insertEnd(list, iteration, wRow-1, wCol-1, rotation);
#ifdef DEBUGMORE
				printf("*** Row = %d, Col = %d\n", wRow-1, wCol-1);
#endif
			}
		}
	}
}

/***********************************************************
  Simple circular linked list for match records
 ***********************************************************/

MATCHLIST* newList()
{
	MATCHLIST* list;

	list = (MATCHLIST*) malloc(sizeof(MATCHLIST));
	if (list == NULL)
		die(__LINE__);

	list->nItem = 0;
	list->tail = NULL;
	list->head = NULL;	

	return list;
}

void deleteList( MATCHLIST* list)
{
	MATCH *cur, *next;
	int i;
	//delete items first

	if (list->nItem != 0 ){
		cur = list->tail->next;
		next = cur->next;
		for( i = 0; i < list->nItem; i++, cur = next, next = next->next ) {
			free(cur); 
		}

	}
	free( list );
}

void insertEnd(MATCHLIST* list, 
		int iteration, int row, int col, int rotation)
{
	MATCH* newItem;

	newItem = (MATCH*) malloc(sizeof(MATCH));
	if (newItem == NULL)
		die(__LINE__);

	newItem->iteration = iteration;
	newItem->row = row;
	newItem->col = col;
	newItem->rotation = rotation;

	if (list->nItem == 0){
		newItem->next = newItem;
		list->tail = newItem;
		list->head = newItem;
	} else {
		newItem->next = list->tail->next;
		list->tail->next = newItem;
		list->tail = newItem;
	}

	(list->nItem)++;

}

void printList(MATCHLIST* list)
{
	int i;
	MATCH* cur;

	printf("List size = %d\n", list->nItem);    


	if (list->nItem == 0) return;

	cur = list->tail->next;
	for( i = 0; i < list->nItem; i++, cur=cur->next){
		printf("%d:%d:%d:%d\n", 
				cur->iteration, cur->row, cur->col, cur->rotation);
	}
}