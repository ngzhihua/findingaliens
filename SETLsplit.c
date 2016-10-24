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

void evolveWorld(char** curWorld, char** nextWorld, int size);

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
	int dir, iterations, iter, rank, numTask, numSlaveProcess, row, tag = 1, rowOffset;
	int size, patternSize;
	long long before, after;
	MATCHLIST*list;
	int *listArr;
	int *resultBuf;

	if (argc < 4 ){
		fprintf(stderr, 
				"Usage: %s <world file> <Iterations> <pattern file>\n", argv[0]);
		exit(1);
	} 


	curW = readWorldFromFile(argv[1], &size);
	nextW = allocateSquareMatrix(size+2, DEAD);


	printf("World Size = %d\n", size);

	iterations = atoi(argv[2]);
	printf("Iterations = %d\n", iterations);

	patterns[N] = readPatternFromFile(argv[3], &patternSize);
	for (dir = E; dir <= W; dir++){
		patterns[dir] = allocateSquareMatrix(patternSize, DEAD);
		rotate90(patterns[dir-1], patterns[dir], patternSize);
	}
	printf("Pattern size = %d\n", patternSize);
	
	list = newList();

#ifdef DEBUG
	printSquareMatrix(patterns[N], patternSize);
	printSquareMatrix(patterns[E], patternSize);
	printSquareMatrix(patterns[S], patternSize);
	printSquareMatrix(patterns[W], patternSize);
#endif


	//Start timer
	before = wallClockTime();

	//	init(&argc, argv, *numTask, *rank);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numTask);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	numSlaveProcess = numTask - 1;	

	for (iter = 0; iter < iterations; iter++){

#ifdef DEBUG
		printf("World Iteration.%d\n", iter);
		printSquareMatrix(curW, size+2);
#endif
		if (rank == 0){
			distributeWork(curW, numSlaveProcess, tag, size, patternSize);
			gatherWork(resultBuf, tag, numTask, list);
			//Generate next generation
			evolveWorld( curW, nextW, size);
			temp = curW;
			curW = nextW;
			nextW = temp;
		}
		else if (rank > 0 && rank < numSlaveProcess){
			//recv
			row = ceil(size / (float)numSlaveProcess) + patternSize + 1;
			//printf("rank != numSlave %d process row :%d\n", rank, row);
			recvBuf = allocateEmptySquareMatrix(row, (size + 2));
			receiveWork(recvBuf, row, size, tag);
			list = newList();
			//printf("process %d started searching\n", rank);
			searchPatterns( recvBuf, size, iter, patterns, patternSize, list, row);
			//printf("process %d finished searching\n", rank);
			//Send back
			rowOffset = (rank-1) * ceil(size / (float)numSlaveProcess);
			listArr = convertMatchListToArr(list, rowOffset);
			//int i;
			//printf("buffered items= %d\n", list->nItem);
			//for (i = 0; i < list->nItem * 4; i++){
			//	printf("%d ", listArr[i]);
			//}
			//printf("process %d returned from converting: %d\n", rank, list->nItem);
			//if (list->nItem > 0){
				//printf("listArr[0] = %d\n", listArr[0]);
				MPI_Send(&(listArr[0]), list->nItem *4, MPI_INT, 0, tag, MPI_COMM_WORLD);
			//}
			if (list != NULL){
			deleteList(list);
			}
			if (listArr != NULL || sizeof(listArr) == 0){
			free(listArr);
			}

		}
		else if (rank == numSlaveProcess){
			//Last process handling is special due to odd sizes
			//recv
			if (size%numSlaveProcess == 0){
				row = size / numSlaveProcess + 2; 
			}
			else{
				row = size - (numSlaveProcess - 1) * ceil(size/(float)numSlaveProcess) + 2;
			}
			//printf("rank == numSlave %d process row : %d\n", rank, row);
			recvBuf = allocateEmptySquareMatrix(row, (size + 2));
			receiveWork(recvBuf, row, size, tag);
			list = newList();
			//printf("process %d started searching\n", rank);
			searchPatterns( recvBuf, size, iter, patterns, patternSize, list, row);
			//printf("process %d finished searching\n", rank);
			//Send back
			rowOffset = (rank-1) * ceil(size / (float)numSlaveProcess);
			listArr = convertMatchListToArr(list, rowOffset);
			//printf("process %d returned from converting: %d\n", rank, list->nItem);
			//if (list->nItem > 0){
			MPI_Send(&(listArr[0]), list->nItem *4, MPI_INT, 0, tag, MPI_COMM_WORLD);
			//	printf("process %d sent to process 0", rank);
			//}
			if (list != NULL){
				deleteList(list);
			}
			if (listArr != NULL || sizeof(listArr) == 0){
				free(listArr);
			}
		}
		tag++;
		MPI_Barrier(MPI_COMM_WORLD);
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
		printList( list );
		
	deleteList( list );
	}
	
	//MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Finalize();

	after = wallClockTime();

	printf("Sequential SETL took %1.2f seconds\n", 
			((float)(after - before))/1000000000);
	//Clean up

	return 0;

}

/***********************************************************
  Helper functions 
 ***********************************************************/
// Receives list and appends to main list
void gatherWork(int *recvBuf, int tag, int numTask, MATCHLIST* list){
	int i, j, recvSize, arrIndex;
	MPI_Status status;
	MATCHLIST *lists[4];
	MATCH *temp;
	//MATCHLIST *list0 = newList();
	//MATCHLIST *list1 = newList();
	//MATCHLIST *list2 = newList();
	//MATCHLIST *list3 = newList();
	//MATCH* curr;
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
			//printf("result received from %d with status %d and count %d and source %d and tag %d\n", i, status.MPI_ERROR, recvSize, status.MPI_SOURCE, status.MPI_TAG);
			
			for (j = 0; j < recvSize ; j += 4){
				//printf("receiveBuffer: %d:%d:%d:%d\n", recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
				//if ( recvBuf[j+3] == 0){
				arrIndex= recvBuf[j+3];
				insertEnd(lists[arrIndex], recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
				//}	
				//else if ( recvBuf[j+3] == 1){
				//	insertEnd(list1, recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
				//}	
				//else if ( recvBuf[j+3] == 2){
				//	insertEnd(list2, recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
				//}
				//else if ( recvBuf[j+3] == 3){
				//	insertEnd(list3, recvBuf[j], recvBuf[j+1], recvBuf[j+2], recvBuf[j+3]);
				//}
				//insertEnd(list, recvBuf[j*4], recvBuf[j*4+1], recvBuf[j*4 +2], recvBuf[j*4+3]);
				//printf("after insertEnd\n");
			}
			//printf("result from %d processed\n", i);
			//free(recvBuf)
		}
	}

	//list->tail = list0->tail;
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
	
	//list->nItem += list0->nItem + list1->nItem + list2->nItem + list3->nItem;
}

int* convertMatchListToArr(MATCHLIST* list, int rowOffset){
	int i;
	int *arr;
	MATCH* curr;
	if (list->nItem > 0){
		curr = list->tail;
	}
	arr = malloc(list->nItem * 4 * sizeof(int));
	//printList(list);
	for ( i= 0 ; i < list->nItem; i++){
		(arr[i*4]) = curr->iteration;
		(arr[i*4+1]) = curr-> row + rowOffset;
		(arr[i*4+2]) = curr-> col;
		(arr[i*4+3]) = curr-> rotation;
		curr = curr->next;
		//printf("%d %d %d %d\n", arr[i*4], arr[i*4+1], arr[i*4+2], arr[i*4+3]);
	}
	return arr;
}

void receiveWork(char **recvBuf, int numRows, int wSize, int tag){
	MPI_Status status;

	MPI_Recv(&(recvBuf[0][0]), numRows * (wSize +2), MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);

	//	printf("count is %d. error is %d.\n", status.count, status.MPI_ERROR);

	//int i, j;

	//for (i = 0; i < numRows; i++){
	//	for (j = 0; j < wSize + 2; j++){
	//		printf("%c", recvBuf[i][j]);
	//	}
	//	printf("\n");
	//}

}

void distributeWork(char ** currW, int numSlaveProcess, int tag, int wSize, int pSize){
	int i, numRows = ceil(wSize / (float)numSlaveProcess) + pSize + 1, rowNum = 0;

	// Distribute to processes 1 to n-1
	for (i = 1; i < numSlaveProcess; i++){
		MPI_Send(&(currW[rowNum][0]), numRows * (wSize + 2), MPI_CHAR, i, tag, MPI_COMM_WORLD);
		//printf("send successful to %d: numRows = %d\n", i, numRows * (wSize +2));
		rowNum += ceil(wSize/(float)numSlaveProcess);
	}

	// Handle last case of odd number of rows seperately
	if (wSize %numSlaveProcess == 0){
		numRows = wSize/numSlaveProcess + 2;
	}
	else{
		numRows = wSize - ((numSlaveProcess - 1) * (ceil(wSize/(float)numSlaveProcess))) + 2 ;
		//printf("number of rows = %d\n", numRows);
	}
	MPI_Send(&(currW[rowNum][0]), numRows * (wSize + 2), MPI_CHAR, numSlaveProcess, tag, MPI_COMM_WORLD);
	//printf("send successful to %d\n", numSlaveProcess);
	tag++;
}

//void init(int* argc, char** argv, int* numTask, int* rank){
//}

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

void evolveWorld(char** curWorld, char** nextWorld, int size)
{
	int i, j, liveNeighbours;

	for (i = 1; i <= size; i++){
		for (j = 1; j <= size; j++){
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

