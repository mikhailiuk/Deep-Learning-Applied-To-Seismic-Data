// Author: Aliaksei Mikhailiuk, 2017.

#include "../headers/DataReader.h"


int DataReader::initialise(ParamsInit parameters){
        m_patchX             = parameters.patchX;
        m_patchY             = parameters.patchY;
        m_patchZ             = parameters.patchZ;
        m_shiftXTest         = parameters.shiftXTest;        
        m_shiftYTest         = parameters.shiftYTest;  
        m_shiftZTest         = parameters.shiftZTest;  
        m_shiftXTrain        = parameters.shiftXTrain;        
        m_shiftYTrain        = parameters.shiftYTrain;  
        m_shiftZTrain        = parameters.shiftZTrain;
        m_augmentX           = (m_patchX>1) ? parameters.augment : 0;
        m_augmentY           = (m_patchY>1) ? parameters.augment : 0;
        m_augmentZ           = (m_patchZ>1) ? parameters.augment : 0;
        m_augmentFlag        = (parameters.augment > 0) ? enabled : disabled;
        m_nameDataTrainIn    = parameters.nameDataTrainIn;
        m_nameDataTestIn     = parameters.nameDataTestIn;
        m_nameMaskTest       = parameters.nameMaskTest; 
        m_nameMaskValidation = parameters.nameMaskValidation; 
        m_flagMaskValidation = (parameters.maskFlagValidation == 0) ? disabled : enabled;  
        m_flagMaskTest       = (parameters.maskFlagTest       == 0) ? disabled : enabled; 
        m_patchSize          = m_patchX*m_patchY*m_patchZ;
        m_inputScale         = parameters.inputScale;
        m_MSError            = 0.0;
        m_qMeasure           = 0.0;
        m_max                = 0.0;
        m_maxAbs             = 0.0;
        m_min                = 0.0;
        m_polishing          = parameters.polishing;
        m_ratioTargetPredictedNeg = 1.0;
        m_ratioTargetPredictedPos = 1.0;
        return 0;
}

int DataReader::readMasks(){


        // set the mask dimentions to the same as train
        m_xWidthMask = m_xWidthTrain;
        m_yWidthMask = m_yWidthTrain;

        // allocate memory for the the validation and test masks
        m_mask.resize(m_xWidthMask);
        m_maskValidation.resize(m_xWidthMask);

        for (int ii = 0; ii < m_xWidthMask; ++ii){
                m_mask[ii].resize(m_yWidthMask);  
                m_maskValidation[ii].resize(m_yWidthMask);       
        }

        // set values of the masks to 1.0 (means no mask), 0.0 means noise (missing value)
        for (int ii = 0; ii < m_xWidthMask; ++ii){
                for (int jj = 0; jj<m_yWidthMask; jj++){
                        m_mask[ii][jj] = 1.0;
                        m_maskValidation[ii][jj] = 1.0;
                }
        }
        

        // if there is a test mask
        if (m_flagMaskTest == enabled){
        
                // Open a file with mask
                std::fstream myfileInMask(m_nameMaskTest, std::ios_base::in);

                if(!myfileInMask) {
                        printf("Error reading test mask\n");
                        exit(0);
                }
                
                // Read the first two values (dimentions)
                myfileInMask >> m_xWidthMask >> m_yWidthMask;

                // Sanity check (whether config values correspond to the data file)
                checkDimensionsMask();

                // Read test mask to a matrix
                for (int ii = 0; ii<m_xWidthMask; ++ii){
                        for (int jj = 0; jj<m_yWidthMask; jj++){
                                myfileInMask>>m_mask[ii][jj];
                 
                        }
                }

                // Close the file
                myfileInMask.close();
        }



        // If using the validation mask
        if (m_flagMaskValidation == enabled){
        
                // Open the validation mask file
                std::fstream myfileInMask(m_nameMaskValidation, std::ios_base::in);

                if(!myfileInMask) {
                        printf("Error reading validation mask\n");
                        exit(0);
                }
                
                // Read the first two values (dimentions)
                myfileInMask >> m_xWidthMask >> m_yWidthMask;

                // Sanity check (whether config values correspond to the data file)
                checkDimensionsMask();

                // Read validation mask to a matrix
                for (int ii = 0; ii < m_xWidthMask; ++ii){
                        for (int jj = 0; jj < m_yWidthMask; ++jj){
                                myfileInMask >> m_maskValidation[ii][jj];
                 
                        }
                }
                myfileInMask.close();
        }


        return 0;
}

int DataReader::readTrainCube(){

        // Open the file with training cube
        std::fstream myfileIn(m_nameDataTrainIn, std::ios_base::in);

        if(!myfileIn) {
                printf("Error reading train in\n");
                exit(0);
        }


        // The first line are the dimentions of the cube
        myfileIn >> m_zWidthTrain >> m_xWidthTrain >> m_yWidthTrain;

        // Check whether the dimentions correspond to those in the config
        checkDimensionsPatch();

        // Allocate the memory for the cube
        m_cube.resize(m_zWidthTrain);

        for (int ii = 0; ii < m_zWidthTrain; ++ii) {

                m_cube[ii].resize(m_xWidthTrain);

                for (int jj = 0; jj < m_xWidthTrain; ++jj){
                        m_cube[ii][jj].resize(m_yWidthTrain);
                }

        }


        // Read the values from the file into the cube variable
        for (int kk = 0; kk < m_zWidthTrain; ++kk) {
                for (int ii = 0; ii < m_xWidthTrain; ++ii) {
                        for (int jj = 0; jj < m_yWidthTrain; ++jj) {
                
                                myfileIn >> m_cube[kk][ii][jj];

                                // Find the maximum absolute value in the file
                                if (std::abs(m_cube[kk][ii][jj]) > m_maxAbs) {
                                        m_maxAbs = std::abs(m_cube[kk][ii][jj]);
                                }

                                // Find the maximum value in the data file
                                if (m_cube[kk][ii][jj] > m_max) {
                                        m_max = m_cube[kk][ii][jj];                             
                                }

                                // Find the minimum value in the file
                                if (m_cube[kk][ii][jj] < m_min) {
                                        m_min = m_cube[kk][ii][jj]; 
                                }
                        }
                }
        }

        myfileIn.close();
        return 0;
}

int DataReader::readDataTrain(vector<double> *dataTrainIn,int *numbIn, int *numbItTrain, vector<double> *maskRearangedTrain, int *xWidthTrain,int *yWidthTrain, vector<double> *maskRearangedValidation){
        

        // Values used to rearage the data cube into patches, identify corner values of the sliding window 
        int startDepth = 0, startXWidth = 0, startYWidth = 0;
        
        // Read the training cube
        readTrainCube();
        
        // Set the scaling cube/m_maxAbs, this way the data is between [-1; 1]
        if (m_inputScale == -1.0){
                m_inputScale = 1.0/m_maxAbs;
        }

        // Read the mask
        readMasks();
 
        // Run through all examples
        for (int ii = 0; ii < (*numbItTrain); ++ii){
               // If the first layer of the cube, rearage the mask patches 
               if(startDepth == 0) {
                        for (int kk = startXWidth; kk < startXWidth+m_patchX; ++kk) {
                                for (int ff = startYWidth; ff < startYWidth+m_patchY; ++ff) {
                                        (*maskRearangedTrain).push_back(m_mask[kk][ff]*(double)m_flagMaskTest);
                                        (*maskRearangedValidation).push_back(m_maskValidation[kk][ff]*(double)m_flagMaskValidation);
                                }
                        }
                }

                // Go through the data cube slice by slice and extract the patch using sliding a window
                for (int jj = startDepth; jj < startDepth+m_patchZ; ++jj) {
                        for (int kk = startXWidth; kk < startXWidth+m_patchX; ++kk) {
                                for (int ff = startYWidth; ff < startYWidth+m_patchY; ++ff) {
                                        
                                        // If the input scale == -2.0, cube = (cube-m_min)/(m_max-m_min)
                                        if (m_inputScale == -2.0){
                                               (*dataTrainIn).push_back((m_cube[jj][kk][ff]-m_min)/(m_max-m_min));  
                                
                                        // Otherwise cube = cube/scale                                     
                                        }else{
                                               (*dataTrainIn).push_back(m_cube[jj][kk][ff]*m_inputScale);
                                        }
                                }                        
                        }                
                }

                // Calculate new indeces for the sliding window (the first value is flag - 1 if the function is called from the training data)
                calculateBounds(1,&startYWidth,&startXWidth,&startDepth,ii,numbItTrain); 
        }

        // Set the values passed back to the training algorithm
        (*xWidthTrain) = m_xWidthTrain; 
        (*yWidthTrain) = m_yWidthTrain;
        (*numbIn)= m_patchSize;
        (*numbItTrain)++;

        // Empty the cube to reuse later
        vector< vector < vector < double > > >().swap(m_cube);
        return 0;

}


int DataReader::readTestCube(){

        // Open the file
        std::fstream myfileIn(m_nameDataTestIn, std::ios_base::in);

        // The first line contains the dimentions
        myfileIn >> m_zWidthTest >> m_xWidthTest >> m_yWidthTest;

        // Allocate the memory
        m_coefCube.resize(m_zWidthTest);
        m_cube.resize(m_zWidthTest);
        for (int ii = 0; ii < m_zWidthTest; ++ii) {
                m_cube[ii].resize(m_xWidthTest);
                m_coefCube[ii].resize(m_xWidthTest);
                for (int jj = 0; jj < m_xWidthTest; ++jj) {
                        m_cube[ii][jj].resize(m_yWidthTest);
                        m_coefCube[ii][jj].resize(m_yWidthTest);
                }
        }       

        // Read the values of the test cube and set to 0.0 all coefficient cube
        for (int kk = 0; kk < m_zWidthTest; ++kk) {
                for (int ii = 0; ii < m_xWidthTest; ++ii) {
                        for (int jj = 0; jj < m_yWidthTest; ++jj) {
                                myfileIn >> m_cube[kk][ii][jj];
                                m_coefCube[kk][ii][jj]=0.0;
                        }
                }
        }

        myfileIn.close();


        return 0;
}


int DataReader::readDataTest(vector<double> *dataTestIn,int *numbItTest,vector<double> *maskRearanged){

        // Values used to rearage the data cube into patches, identify corner values of the sliding window 
        int startDepth = 0, startXWidth = 0, startYWidth = 0;
	

        // Read the test cube
        readTestCube();

        // Run through all examples
        for (int ii = 0; ii < (*numbItTest); ++ii){

                // If the first layer of the cube, rearage the mask patches 
               if(startDepth == 0){
                        for (int kk = startXWidth; kk < startXWidth+m_patchX; ++kk){
                                for (int ff = startYWidth; ff<startYWidth+m_patchY; ++ff){
                                        (*maskRearanged).push_back(m_mask[kk][ff]*(double)m_flagMaskTest);
                                }
                        }
                }

                // Go through the data cube slice by slice and extract the patch using sliding a window
                for (int jj = startDepth; jj < startDepth+m_patchZ; ++jj){
                        for (int kk = startXWidth; kk < startXWidth+m_patchX; ++kk){
                                for (int ff = startYWidth; ff < startYWidth+m_patchY; ++ff){

                                        // If the input scale == -2.0, cube = (cube-m_min)/(m_max-m_min)
                                        if (m_inputScale == -2.0){
                                               (*dataTestIn).push_back((m_cube[jj][kk][ff]-m_min)/(m_max-m_min));
                                        
                                        // Otherwise cube = cube/scale 
                                        }else{
                                               (*dataTestIn).push_back(m_cube[jj][kk][ff]*m_inputScale);
                                        }
                                }                        
                        }                
                }


                // Calculate new indeces for the sliding window (the first value is flag - 0 if the function is called from the test data)
                calculateBounds(0,&startYWidth,&startXWidth,&startDepth,ii,numbItTest);       
        }

        (*numbItTest)++;
        m_numbItTest= (*numbItTest);

        // Set the values of the coefficient cube
        calculateCoefficientCube();
        
        return 0;

}


int  DataReader::calculateBounds(int trainFlag, int *startYWidth, int *startXWidth, int *startDepth, int ii, int *numbIterations){

        // Set the values to which the boundaries of the sliding window is compered to the boundaries of the data cubes
        int valX = m_shiftXTest*(1-trainFlag)+m_shiftXTrain*trainFlag;
        int valY = m_shiftYTest*(1-trainFlag)+m_shiftYTrain*trainFlag;
        int valZ = m_shiftZTest*(1-trainFlag)+m_shiftZTrain*trainFlag;
        int compareValueY = m_yWidthTest*(1-trainFlag)+m_yWidthTrain*trainFlag;
        int compareValueX = m_xWidthTest*(1-trainFlag)+m_xWidthTrain*trainFlag;
        int compareValueZ = m_zWidthTest*(1-trainFlag)+m_zWidthTrain*trainFlag;


        // If the (starting point in Y dimention + shift + Y dimention of the patach) less than the boundary in Y
        if ((*startYWidth)+valY+m_patchY <= compareValueY){

                // Shift the starting value
               (*startYWidth) += valY;

        } else {
        
                // Start at 0 in the Y dimention of the cube
                (*startYWidth) = 0;

                // If the (starting point in X dimention + shift + X dimention of the patach) less than the boundary in X
                if ((*startXWidth)+valX+m_patchX <= compareValueX){

                        // Shift the starting value
                        (*startXWidth) += valX;
                } else {

                        // Start at 0 in the X dimention of the cube
                        (*startXWidth) = 0;

                        // If the (starting point in Z dimention + shift + Z dimention of the patach) less than the boundary in Z
                        if ((*startDepth)+valZ+m_patchZ <= compareValueZ){

                                // Start at 0 in the Z dimention of the cube
                                (*startDepth) += valZ;

                        // If reached this state then reached the end of the data cube, set the counter to the max number of iterations
                        } else {
                                
                                (*numbIterations) = ii;
                        }
                }
       }

       // The last example makes the data to shift and brings inconsistency, do not update 
       if (ii == (*numbIterations)-1) (*numbIterations)--;

       return 0;
}




double DataReader::calculateCoefficientCube(){

        // Values for the starting positions of the sliding window
        int startZ = 0, startX = 0, startY = 0;

        //Go through all the patches
        for (int ii = 0; ii<m_numbItTest; ++ii){
                for (int jj = startZ + m_augmentZ; jj<startZ + m_patchZ - m_augmentZ; ++jj){
                        for (int kk = startX + m_augmentX; kk < startX + m_patchX - m_augmentX; ++kk){
                                for (int ff = startY + m_augmentY; ff < startY + m_patchY - m_augmentY; ++ff){

                                        // Add 1 for every patch at the position [jj][kk][ff] (gives the number of overlapping patches)
                                        m_coefCube[jj][kk][ff]+=1.0;
 
                                }
                        }
                }
                calculateBounds(0,&startY,&startX,&startZ,ii,&m_numbItTest);
        }


        // Those values that are left 0 should be set to 1 (since when the cube is written cube = cube/coefCube)
        for (int jj = 0; jj < m_zWidthTest; ++jj) {
		for (int ii = 0; ii < m_xWidthTest; ++ii) {
                        for (int kk = 0; kk < m_yWidthTest; ++kk) {
                                if (m_coefCube[jj][ii][kk]==0.0){
                                        m_coefCube[jj][ii][kk] = 1.0;
                                }

                        }
		}
	}


        return 0.0;

}



double DataReader::calculateSNR(vector<double> dataClean,vector<double> dataNoize,std::string path){

        // The procefure follows: 10 log (|x_t|/|x_t-x_p|) where x_t - target, x_p - predicted

        double qDenom = 0.0;
        double qNum = 0.0;

        double clean, noise;
        int maskCounter = 0;
        int finZ, finY, finX;
        int counter = 0, startZ = 0, startX = 0, startY = 0;


        // Create two cubes which hold the predicted and the target data
        vector< vector < vector < double > > > cubeClean;
        vector< vector < vector < double > > > cubeNoisy;

        // Allocate the memory for the cubes        
        cubeClean.resize(m_zWidthTest);
        cubeNoisy.resize(m_zWidthTest);
        for (int ii = 0; ii < m_zWidthTest; ++ii) {
                cubeClean[ii].resize(m_xWidthTest);
                cubeNoisy[ii].resize(m_xWidthTest);
                for (int jj = 0; jj < m_xWidthTest; ++jj){
                        cubeClean[ii][jj].resize(m_yWidthTest);
                        cubeNoisy[ii][jj].resize(m_yWidthTest);
                }
        } 

        // Go through the test examples and copy the data from training arrays to the data cubes, scaling by the number of overlapping patches
        for (int ii = 0; ii < m_numbItTest; ++ii){ 
                for (int jj = startZ + m_augmentZ; jj<startZ + m_patchZ - m_augmentZ; ++jj){
                        for (int kk = startX + m_augmentX; kk < startX + m_patchX - m_augmentX; ++kk){
                                for (int ff = startY + m_augmentY; ff < startY + m_patchY - m_augmentY; ++ff){
                                        counter = ii*m_patchX*m_patchY*m_patchZ + (jj - startZ)*m_patchX*m_patchY + (kk - startX)*m_patchY + (ff -  startY);
                                        cubeNoisy[jj][kk][ff] += dataNoize[counter]/m_coefCube[jj][kk][ff];
                                        cubeClean[jj][kk][ff] += dataClean[counter]/m_coefCube[jj][kk][ff];
                                }
                        }
                }
                finZ = startZ + m_patchZ - m_augmentZ-1;
                finY = startY + m_patchY - m_augmentY-1;
                finX = startX + m_patchX - m_augmentX-1;
                calculateBounds(0,&startY,&startX,&startZ,ii,&m_numbItTest);
        }



        // The loop to bring the data back to its original domain, by applying inverse scaling
	for (int jj = m_augmentZ; jj < m_zWidthTest-m_augmentZ; ++jj) {
		for (int ii = m_augmentX; ii < m_xWidthTest-m_augmentX; ++ii) {
                        for (int kk = m_augmentY; kk < m_yWidthTest-m_augmentY; ++kk) {
                                
                                //Apply the inverse scaling 
                                if (m_inputScale == -2.0) {
                                        clean = cubeClean[jj][ii][kk]*(m_max-m_min)+m_min;
                                        noise = cubeNoisy[jj][ii][kk]*(m_max-m_min)+m_min;
                                } else {                                      
                                        clean = cubeClean[jj][ii][kk]/m_inputScale;
                                        noise = cubeNoisy[jj][ii][kk]/m_inputScale;
                                }

                                if (finZ==jj && finY==kk && ii==finX){
                                        // want to finish calculating the error if reached the end of the cube
                                        jj = m_zWidthTest-m_augmentZ;
                                        kk = m_yWidthTest-m_augmentY;
                                        ii = m_xWidthTest-m_augmentX;                    
                                }else{
                                        // qNum = |x_t|, only for the missing values (achieved with: 1.0 - m_mask[ii][kk])
                                        qNum += clean*clean*(1.0 - m_mask[ii][kk]);


                                        // qDen = |x_p - x_t|, only for the missing values
                                        qDenom += (clean-noise)*(clean-noise)*(1.0 - m_mask[ii][kk]);

                                        
                                        // Calculate the number of missing values 
                                        maskCounter+=static_cast<int>(1.0 - m_mask[ii][kk]);
                                }



                        }
		}
	}

        // MSE = qDen/Number_of_values
        m_MSError = qDenom/static_cast<double>(maskCounter);

        //10 log (|x_t|/|x_t-x_p|)
        m_qMeasure = 10*log10(qNum/qDenom);

        //Write the error results to a file
	std::ofstream write_file;
	write_file.open(path,  std::ios_base::out);
        write_file<<m_MSError<<"\n"<<m_qMeasure<<"\n";
	write_file.close();

        return 0.0;

}



int DataReader::writeDataCube(vector<double> data,std::string path, double flagMask){

        // Starting position of the sliding window
        int counter = 0, startZ = 0, startX = 0, startY = 0;


        // Set all values of the cube to 0
        zeroCube();


        // Copy the training data into a cube
        for (int ii = 0; ii < m_numbItTest; ++ii){ 
                for (int jj = startZ+m_augmentZ; jj<startZ+m_patchZ-m_augmentZ; ++jj){
                        for (int kk = startX+m_augmentX; kk < startX+m_patchX-m_augmentX; ++kk){
                                for (int ff = startY+m_augmentY; ff < startY+m_patchY-m_augmentY; ++ff){
                                        counter = ii*m_patchX*m_patchY*m_patchZ + (jj - startZ)*m_patchX*m_patchY + (kk - startX)*m_patchY + (ff -  startY);
                                       
                                        m_cube[jj][kk][ff] += data[counter]/m_coefCube[jj][kk][ff]*m_mask[kk][ff]*(flagMask)+data[counter]/m_coefCube[jj][kk][ff]*(1-flagMask);

                                }
                        }
                }
                calculateBounds(0,&startY,&startX,&startZ,ii,&m_numbItTest);
        }
        m_numbItTest++;

        writeCubeIntoFile(path,flagMask);

        return 0;
}


int DataReader::writeDataCubePrediction(vector<double> dataTarget,vector<double> dataPred,std::string path){

        // Starting position of the sliding window
        int counter = 0, startZ = 0, startX = 0, startY = 0;

        // Set all values of the cube to 0
        zeroCube();

        // Go through the test examples
        for (int ii = 0; ii < m_numbItTest; ++ii) { 

                // Go through the data with a sliding window
                for (int jj = startZ+m_augmentZ; jj<startZ+m_patchZ-m_augmentZ; ++jj) {
                        for (int kk = startX + m_augmentX; kk < startX + m_patchX - m_augmentX; ++kk) {
                                for (int ff = startY+m_augmentY; ff < startY + m_patchY-m_augmentY; ++ff) {

                                        // Calculate the index in the data array
                                        counter = ii*m_patchX*m_patchY*m_patchZ + (jj - startZ)*m_patchX*m_patchY + (kk - startX)*m_patchY + (ff -  startY);
                                        // For the missing values write the predicted, for initial values write the target values
                                        m_cube[jj][kk][ff] += dataTarget[counter]/m_coefCube[jj][kk][ff]*m_mask[kk][ff]+(1-m_mask[kk][ff])*dataPred[counter]/m_coefCube[jj][kk][ff];
                                        
                                }                        
                        }
                }
                calculateBounds(0,&startY,&startX,&startZ,ii,&m_numbItTest);
        }

        m_numbItTest++;

        writeCubeIntoFile(path,0.0);

        return 0;

}

double DataReader::calculateAverageDataPredictedTarget(){

        double sumTargetPos = 0.0,sumTargetNeg = 0.0, sumPredictedPos = 0.0,sumPredictedNeg = 0.0;
        int numberOfTargetPos=0,numberOfTargetNeg=0, numberOfPredictedPos = 0, numberOfPredictedNeg = 0;
        // Write scaled values
	for (int jj = 0; jj < m_zWidthTest; ++jj) {
		for (int ii = 0; ii < m_xWidthTest; ++ii) {
                        for (int kk = 0; kk < m_yWidthTest; ++kk) {
                                if (m_mask[ii][kk]==0.0){

                                        if (m_cube[jj][ii][kk]>0){
                                                sumPredictedPos+=fabs(m_cube[jj][ii][kk]);
                                                numberOfPredictedPos++;                                                
                                        }else{
                                                sumPredictedNeg+=fabs(m_cube[jj][ii][kk]);
                                                numberOfPredictedNeg++;
                                        }

                                }else{
                                        if (m_cube[jj][ii][kk]>0){
                                                sumTargetPos+=fabs(m_cube[jj][ii][kk]);
                                                numberOfTargetPos++;                                                
                                        }else{
                                                sumTargetNeg+=fabs(m_cube[jj][ii][kk]);
                                                numberOfTargetNeg++;
                                        }


                                }
                        }
		}

	}

        m_ratioTargetPredictedPos = fabs((sumTargetPos/static_cast<double>(numberOfTargetPos))/(sumPredictedPos/static_cast<double>(numberOfPredictedPos)));
        m_ratioTargetPredictedNeg = fabs((sumTargetNeg/static_cast<double>(numberOfTargetNeg))/(sumPredictedNeg/static_cast<double>(numberOfPredictedNeg)));
}


void DataReader::writeCubeIntoFile(std::string path, double maskFlag){

        double ratio;
        // Create the file
	std::ofstream write_file;
	write_file.open(path,  std::ios_base::out);

        //Write the dimentions first
        write_file << m_zWidthTest << " "
                   << m_yWidthTest << " "
                   << m_xWidthTest;
        write_file << "\n";

        // Write scaled values
	for (int jj = 0; jj < m_zWidthTest; ++jj) {
		for (int ii = 0; ii < m_xWidthTest; ++ii) {
                        for (int kk = 0; kk < m_yWidthTest; ++kk) {

                                // Apply the mapping to the initial data domain 
                                if (m_inputScale == -2.0){
                                        
                                        // cube = cube*(max-min)+min
                                        double value =(m_cube[jj][ii][kk]*(m_max-m_min)+m_min)*(1-maskFlag)+(m_cube[jj][ii][kk]*(m_max-m_min)+m_min)*m_mask[ii][kk]*maskFlag;

                                        write_file << value << " ";

                                                                
                                } else {

                                        // cube = cube/scale
                                        write_file << (m_cube[jj][ii][kk]/m_inputScale) << " ";


                                }
                        }
		}
		write_file << "\n";
	}
	write_file.close();
}


void DataReader::zeroCube(){

        // Set the cube to 0
        for (int ii = 0; ii < m_zWidthTest; ++ii) {
                for (int jj = 0; jj < m_xWidthTest; ++jj) {
                        for (int kk = 0; kk < m_yWidthTest; ++kk) {
                                m_cube[ii][jj][kk] = 0.0;
                        }
                }
        }
}


int DataReader::checkDimensionsPatch(){

        // Cannot have the size of the patch larger than the data dimentions

        if (m_patchX > m_xWidthTrain){
                printf("Error: Patch x and input x dimensions mismatch\n");
                exit(0);
        }

        if (m_patchZ > m_zWidthTrain){
                printf("Error: Patch z and input z dimensions mismatch\n");
                exit(0);
        }

        if (m_patchY > m_yWidthTrain){
                printf("Error: Patch y and input y dimensions mismatch\n");
                exit(0);
        }
        return 0;
}

int DataReader::checkDimensionsMask(){

        // Mask dimentions must be the same as of the data 
        if (m_xWidthMask != m_xWidthTrain){
                printf("Mask-data x dimensions mismatch\n");
                exit(0);
        }      
    
        if (m_yWidthMask != m_yWidthTrain){
                printf("Mask-data y dimensions mismatch\n");
                exit(0);
        }
        return 0;
}



