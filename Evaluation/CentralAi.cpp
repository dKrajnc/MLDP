#include <CentralAi.h>
#include <Evaluation/NelderMeadOptimizer.h>
#include <Evaluation/PipelineModel.h>
#include <Evaluation/PipelineAnalytics.h>

namespace dkeval
{


//-----------------------------------------------------------------------------

void CentralAi::execute()
{	
	//Initialize population
	initializePopulation( mOffspringCount );

	//Iterate population in order to find the fittest pipeline + hyperparameter combination over training data
	for ( int i = 0; i < mIterationCount; ++i )
	{
		iteratePopulation();	
	}

	//Validate models
	evaluatePopulation();
}

//-----------------------------------------------------------------------------

void CentralAi::iteratePopulation()
{
	Population offsprings;

	int populationSize                = mPopulation.size();
	int attemptToCreateOffspringMax   = populationSize;
	int attemptToCreateOffspringCount = 0;

	while ( offsprings.size() < mPopulation.size() )
	{
		if ( attemptToCreateOffspringCount > attemptToCreateOffspringMax )
		{
			//Extreme the mutation rate
			mMutationRate = 0.6;
		}

		auto parents          = this->parents( mPopulation );
		auto offspring        = this->offspring( parents.first, parents.second );
		auto isValidOffspring = mTree->isValidPath( offspring );

		if ( !isValidOffspring )
		{
			qDebug() << "Error - Not valid offspring!";

			std::exit( EXIT_SUCCESS );
		}
		else
		{
			if ( mPopulation.values().contains( offspring ) )  // Is Offspring a clone?
			{
				++attemptToCreateOffspringCount;
			}
			else
			{
				double fitness = calculateFitness( offspring, true );

				attemptToCreateOffspringCount = 0;
				offsprings.insertMulti( fitness, offspring );
			}
		}
	}

	// Merge population and ofsprings and clear ofsprings
	Population mergedPopulation;
	mergedPopulation = mPopulation;

	for ( int i = 0; i < offsprings.size(); ++i )
	{
		auto key   = offsprings.keys().at( i );
		auto value = offsprings.values().at( i );

		mergedPopulation.insertMulti( key, value );
	}

	// Remove N least fitt creatures.
	mPopulation.clear();

	for ( int i = 0; i < populationSize; ++i )
	{
		auto key   = mergedPopulation.keys().at( i );
		auto value = mergedPopulation.values().at( i );

		mPopulation.insertMulti( key, value );
	}	    	
}

//-----------------------------------------------------------------------------

double CentralAi::calculateFitness( Creature aCreature, bool aIsTraining )
{
	if ( aIsTraining )
	{
		auto pipelineModel     = new dkeval::PipelineModel( mSettings, aCreature, mTrainingData );
		pipelineModel->setFoldId( mFoldId );

		auto pipelineAnalytics = new dkeval::PipelineAnalytics( mSettings, &mTrainingData, pipelineModel );
		auto inputCount        = pipelineModel->inputCount(); //number of parameters

		//Generate initial and scale vectors for NelderMeadOptimizer constructor arguments
		QVector< double > init;
		init.resize( inputCount );
		init.fill( 0.0 );
		init[ 0 ] = 1.0;

		QVector< double > scale;
		scale.resize( inputCount );
		scale.fill( 10.0 );

		//Optimize parameters with Nelder-Mead
		auto optimizer = lpmleval::NelderMeadOptimizer( pipelineModel, pipelineAnalytics, init, scale, 0.00001, 100, true );		
		optimizer.build();	

		auto fitness = pipelineModel->fitness(); // ROC distance 	

		if ( pipelineModel->dpactions().isEmpty() )
		{
			qDebug() << "Error - DPActions are empty!";			
		}

		
		//Store fittest model information 		
		if ( mPopulation.isEmpty() ) //initial creature evaluation 
		{
			if ( !mTBPAction.isEmpty() )
			{
				qDebug() << "Warning - mTBPAction is not empty in initialization!!!";

				//Clear TBPAaction in CentralAI 
				for ( auto& action : mTBPAction )
				{
					action = nullptr;
				}
				mTBPAction.clear();
			}

			if ( fitness <= 0.1 )
			{
				pipelineAnalytics->setDataPackage( &mTrainingData ); //Apply preprocessing steps	
				mTBPAction = pipelineModel->dpactions(); //best pre-processing algorithm pipeline

				dkeval::PreprocessedPackage preprocessedData;
				preprocessedData.preprocessedDataPackage = pipelineAnalytics->preProcessedDataPackage();
				preprocessedData.tbpActions              = mTBPAction;

				mPreprocessedDatasets.push_back( preprocessedData );
			}			
		}
		else if ( mPopulation.firstKey() == fitness || fitness < mPopulation.firstKey() )
		{
			//Clear TBPAaction in CentralAI 
			for ( auto& action : mTBPAction )
			{
				action = nullptr;
			}
			mTBPAction.clear();

			pipelineAnalytics->setDataPackage( &mTrainingData ); //Apply preprocessing steps	
			mTBPAction = pipelineModel->dpactions(); //best pre-processing algorithm pipeline

			dkeval::PreprocessedPackage preprocessedData;
			preprocessedData.preprocessedDataPackage = pipelineAnalytics->preProcessedDataPackage();
			preprocessedData.tbpActions = mTBPAction;

			mPreprocessedDatasets.push_back( preprocessedData );			
		}	

		delete pipelineModel;
		delete pipelineAnalytics;

		return fitness;
	}
	else
	{		
		qDebug() << "Warning - evaluating non-training data!";
	}	
}

//-----------------------------------------------------------------------------

Creature CentralAi::offspring( Creature aParent_1, Creature aParent_2 )
{
	QString chosenParent;
	Creature nodePath;

	//Check if parents are valid
	auto isValidParent_1 = mTree->isValidPath( aParent_1 );
	auto isValidParent_2 = mTree->isValidPath( aParent_2 );

	if ( !isValidParent_1 || !isValidParent_2 )
	{
		qDebug() << "Error - Not valid parents in the offspring";

		std::exit( EXIT_SUCCESS );
	}	

	auto root           = mTree->treeRoot();	
	auto algorithms_1   = mTree->algorithmNames( aParent_1 );
	auto algorithms_2   = mTree->algorithmNames( aParent_2 );	
	auto siblings       = mTree->children( root );	
	int chromosomeIndex = 0;
	
	while ( true )
	{
		int momChromosomeIndex = aParent_1.size() > chromosomeIndex ? chromosomeIndex : -1;
		int dadChromosomeIndex = aParent_2.size() > chromosomeIndex ? chromosomeIndex : -1;	
		auto mom               = momChromosomeIndex == -1 ? nullptr : aParent_1.at( chromosomeIndex );
		auto dad               = dadChromosomeIndex == -1 ? nullptr : aParent_2.at( chromosomeIndex );

		if ( mom != nullptr || dad != nullptr )
		{
			std::shared_ptr< Node > offspringNode = nullptr;

			if ( mom != nullptr && dad != nullptr )
			{				
				chosenParent = this->chooseParent();
			}
			else if ( mom != nullptr )
			{
				chosenParent = "Mom";
			}
			else if ( dad != nullptr )
			{
				chosenParent = "Dad";
			}
			else
			{
				qDebug() << "Impossible!";
			}


			if ( chosenParent == "Mom" )
			{
				offspringNode = mom;
			}
			else if ( chosenParent == "Dad" )
			{
				offspringNode = dad;
			}	


			offspringNode = mutate( offspringNode, siblings );			
			nodePath.push_back( offspringNode );


			if ( mTree->isLeaf( offspringNode ) )
			{			
				break;
			}
			else
			{
				siblings = mTree->children( offspringNode );
			}					
		}
		else
		{			
			std::shared_ptr< Node > offspringNode = mutate( nullptr, siblings );
			nodePath.push_back( offspringNode );			


			if ( mTree->isLeaf( offspringNode ) )
			{
				break;
			}
			else
			{
				siblings = mTree->children( offspringNode );
			}			
		}				

		++chromosomeIndex; 
	}

	return nodePath; 
}

//-----------------------------------------------------------------------------

QString CentralAi::chooseParent()
{
	auto parentChoiceDice = randomPercentage();

	if ( parentChoiceDice > 0.50 )
	{
		return "Mom";
	}
	else
	{
		return "Dad";
	}
}

//-----------------------------------------------------------------------------

QString CentralAi::chooseChild( QList< QString >& aChildren )
{
	QString child;

	if ( aChildren.isEmpty() == true )
	{
		return "";
	}
	else if ( aChildren.size() == 1 )
	{
		return aChildren.at( 0 );
	}
	else
	{
		auto index = randomIndex( aChildren.size() );

		return aChildren.at( index );
	}
}

//-----------------------------------------------------------------------------

double CentralAi::randomPercentage()
{
	std::uniform_real_distribution< double > dice( 0.0, 1.0 );

	return dice( *mRng );
}

//-----------------------------------------------------------------------------

int CentralAi::randomIndex( int aListSize )
{
	std::uniform_int_distribution< int > dice( 0, aListSize - 1 );

	return dice( *mRng );
}

//-----------------------------------------------------------------------------

std::shared_ptr< Node > CentralAi::nodeIfAlgContains( QVector< std::shared_ptr< Node > >& aSiblings, std::shared_ptr< Node > aNode )
{
	for ( auto sibling : aSiblings )
	{
		QString nodeName = aNode == nullptr ? "NA" : aNode->element;

		if ( sibling->element == nodeName )
		{
			return sibling;
		}
	}

	return nullptr;
}

//-----------------------------------------------------------------------------

std::shared_ptr< Node > CentralAi::mutate( std::shared_ptr< Node > aOffspring, QVector< std::shared_ptr< Node > >& aSiblings )
{
	if ( aSiblings.isEmpty() )
	{
		qDebug() << "Error - siblings is empty in mutate! ";
	}

	auto mutationChance = randomPercentage();

	if ( aOffspring == nullptr )
	{
		auto index = randomIndex( aSiblings.size() );

		return aSiblings.at( index );
	}
	else
	{
		auto nodeInSiblings = nodeIfAlgContains( aSiblings, aOffspring );

		if ( nodeInSiblings == nullptr )
		{
			auto index = randomIndex( aSiblings.size() );

			return aSiblings.at( index );
		}

		if ( mutationChance < mMutationRate )  // Mutation
		{
			QVector< std::shared_ptr< Node > > otherSiblings;
			for ( auto sibling : aSiblings )
			{
				if ( sibling != nodeInSiblings )
				{
					otherSiblings.push_back( sibling );
				}
			}

			if ( otherSiblings.isEmpty() )  // No other siblings.
			{
				return nodeInSiblings;
			}
			else
			{
				auto index = randomIndex( otherSiblings.size() );

				return otherSiblings.at( index );
			}
		}
		else
		{
			return nodeInSiblings;
		}
	}

	return nullptr;
}

//-----------------------------------------------------------------------------

void CentralAi::removeIfContains( QList< QString >& aList, QString aElement )
{
	if ( aList.contains( aElement ) )
	{
		auto index = aList.indexOf( aElement );

		aList.removeAt( index );
	}
}

//-----------------------------------------------------------------------------

void CentralAi::initializePopulation( const int& aNumberOfCreatures ) 
{
	for ( int i = 0; i < aNumberOfCreatures; ++i )
	{
		auto randomCreature = mTree->randomPath();
		if ( randomCreature.isEmpty() )
		{
			qDebug() << "Warning - randomCreature is empty!";

			std::exit( EXIT_SUCCESS );
		}

		double fitness = calculateFitness( randomCreature, true );

		mPopulation.insertMulti( fitness, randomCreature ); //insert into mPopulation directly
	}
}

//-----------------------------------------------------------------------------

QPair< Creature, Creature > CentralAi::parents( const Population& aPopulation )
{
	QPair< Creature, Creature > parents;
	Population group_1;
	Population group_2;

	auto shuffledCreatures = aPopulation.values();

	std::random_shuffle( shuffledCreatures.begin(), shuffledCreatures.end() );

	for ( int i = 0; i < shuffledCreatures.size(); ++i )
 	{
		auto shuffledCreature = shuffledCreatures.at( i );
		auto keyOfCreature    = aPopulation.key( shuffledCreature );

		if ( i % 2 == 0 )
		{
			group_1.insertMulti( keyOfCreature, shuffledCreature );
		}
		else
		{
			group_2.insertMulti( keyOfCreature, shuffledCreature );
		}
	}

	parents.first  = group_1.values().at( 0 );
	parents.second = group_2.values().at( 0 );

	return parents;
}

//-----------------------------------------------------------------------------

lpmldata::DataPackage CentralAi::preProcessData( const lpmldata::DataPackage& aData )
{
	lpmldata::DataPackage preprocessedData = aData;

	for ( auto& algorithm : mTBPAction )
	{		
		preprocessedData = algorithm->run( preprocessedData );		
	}

	return preprocessedData;
}

//-----------------------------------------------------------------------------

void CentralAi::saveFoldAt( QString aFoldPath )
{
	lpmlfio::TabularDataFileIo fileIo;	

	fileIo.save( aFoldPath + "/TDS.csv", mPreprocessedDataPackage.featureDatabase() );
	fileIo.save( aFoldPath + "/TLD.csv", mPreprocessedDataPackage.labelDatabase() );
	fileIo.save( aFoldPath + "/VDS.csv", mValidationData.featureDatabase() );
	fileIo.save( aFoldPath + "/VLD.csv", mValidationData.labelDatabase() );
}

//-----------------------------------------------------------------------------

void CentralAi::savePipelineInfo( QString aFoldPath, QString aFileName )
{
	std::string filePath = aFoldPath.toStdString() + aFileName.toStdString();

	std::ofstream textFile;
	textFile.open( filePath, std::fstream::app );

	textFile << "Fold parameters:";
	textFile << "\n";
	textFile << "\n";

	//Apply feature space algorithms from established pipeline
	for ( auto action : mTBPAction )
	{
		if ( mTBPAction.isEmpty() )
		{
			qDebug() << "savePipelineInfo:" << "pipeline in fold" << mFoldId << "is empty!";
		}

		//Store action parameters in a file 		
		auto parameters = action->parameters();

		for ( int i = 0; i < parameters.size(); ++i )
		{
			auto key = parameters.keys().at( i );
			textFile << key.toStdString() << "=" << parameters.value( key ).toString().toStdString();
			textFile << "\n";
		}
		textFile << "\n";		
	}
	textFile << "\n";

	textFile << "Action pipeline:";
	textFile << "\n";
	textFile << "\n";

	//Print algorithm order in pipeline
	for ( auto action : mTBPAction )
	{
		int counter = 1;		
		textFile << QString::number( counter ).toStdString() << "->" << action->id().toStdString() << "\n";
		counter++;
	}


	textFile.close();
}

//-----------------------------------------------------------------------------

void CentralAi::savePerformanceInfo( QString aFoldPath, QString aFileName )
{
	auto filePath = aFoldPath.append( aFileName );
	QFile file( filePath );
	if ( file.open( QIODevice::WriteOnly | QIODevice::Text ) )
	{
		// We're going to streaming text to the file
		QTextStream stream( &file );

		int TP = 0;
		int TN = 0;
		int FP = 0;
		int FN = 0;

		for ( int i = 0; i < mConfusionMatrixValues.size(); ++i )
		{
			auto elementKey = mConfusionMatrixValues.keys().at( i );
			auto elementValue = mConfusionMatrixValues.value( elementKey );

			if ( elementKey == "TP" )
			{
				TP = elementValue;
			}
			else if ( elementKey == "TN" )
			{
				TN = elementValue;
			}
			else if ( elementKey == "FP" )
			{
				FP = elementValue;
			}
			else if ( elementKey == "FN" )
			{
				FN = elementValue;
			}

		}

		dkeval::CMAnalytics cmAnalytics( mConfusionMatrixValues );

		stream << "File path: ;" << filePath << endl;
		stream << endl;
		stream << endl;
		stream << endl;
		stream << ";" << ";" << "TP;" << "TN;" << "FP;" << "FN;" << "ROC;" << endl;
		stream << ";" << ";" << TP << ";" << TN << ";" << FP << ";" << FN << ";" << mROC << ";" << "\n";
		stream << endl;
		stream << "ACC;" << cmAnalytics.acc() << endl;;
		stream << "SNS;" << cmAnalytics.sns() << endl;;
		stream << "SPC;" << cmAnalytics.spc() << endl;;
		stream << "NPV;" << cmAnalytics.npv() << endl;;
		stream << "PPV;" << cmAnalytics.ppv() << endl;;
		stream << endl;;


		file.close();
	}
	else
	{
		qDebug() << "Error, no results saved!";
	}
}

//-----------------------------------------------------------------------------

void CentralAi::evaluatePopulation()
{
	QMap< double, int > results;
	QList< lpmldata::DataPackage > validationSets;
	QVector< QVector< QPair< int, int > > > allValidationIndices;

	for ( int i = 0; i < mPreprocessedDatasets.size(); ++i )
	{
		auto pipeline                = mPreprocessedDatasets.at( i ).tbpActions;
		auto preprocessedDataPackage = *mPreprocessedDatasets.at( i ).preprocessedDataPackage;
		
		//Train the plugin model over pre-processed training data
		auto model     = new lpmleval::RandomForestModel( mPluginSettings );
		auto analytics = new lpmleval::ConfusionMatrixAnalytics( mSettings, &preprocessedDataPackage );
		auto optimizer = new lpmleval::RandomForestOptimizer( mPluginSettings, &preprocessedDataPackage, model, analytics );
		
		optimizer->build();
		
		//---------------------------------------------------------------------------------------------------------------------		

		//Apply feature space algorithms from established pipeline on validation data
		auto validationData = mValidationData;

		if ( !pipeline.isEmpty() )
		{
			for ( auto action : pipeline )
			{
				if ( action->id() == "FS" || action->id() == "PCA" )
				{
					validationData = action->run( mValidationData );
				}
			}
		}
		else
		{
			qDebug() << "Warning - isValidation:" << "pipeline in fold" << mFoldId << "is empty!";
		}

		
		//Validate the trained model over validation data
		analytics   = nullptr;
		analytics   = new lpmleval::ConfusionMatrixAnalytics( mSettings, &validationData );
		auto result = analytics->evaluate( model );	//evaluate training model over validation data; return the ROC 


		//Get individual prediction for each sample to generate AUC
		QVector< QPair< int, int > > validationIndices;

		for ( auto key : validationData.sampleKeys() )
		{
			QVariantList featureVectorVariant = validationData.featureDatabase().value( key );
			QVariantList labelVariant         = validationData.labelDatabase().value( key );		
			
			QVector< double > featureVector;
			for ( int i = 0; i < featureVectorVariant.size(); ++i )
			{
				featureVector.push_back( featureVectorVariant.at( i ).toDouble() );
			}
		
			QVariant evalautedLabel = model->evaluate( featureVector );
			QVariant originalLabel  = labelVariant.at( validationData.labelIndex() );		
			int evaluatedIndex      = validationData.labelOutcomes().indexOf( evalautedLabel.toString() );
			int originalIndex       = validationData.labelOutcomes().indexOf( originalLabel.toString() );

			
			//store the evalated and original indices
			QPair< int, int > indices(evaluatedIndex, originalIndex);
			validationIndices.push_back( indices );
		}
		allValidationIndices.push_back( validationIndices );
		

		//Store confusion matrix values of a given fold
		if ( mConfusionMatrixValues.isEmpty() || result < results.first() )
		{
			//store only for best performing
			mConfusionMatrixValues.clear();
			mConfusionMatrixValues = analytics->confusionMatrixElements(); //TP, TN, FP, FN					
		}
		
		results.insertMulti( result, i );
		validationSets.push_back( validationData );

		delete model;
		delete analytics;
		delete optimizer;
	}

	auto index               = results.values().at( 0 );
	mROC                     = results.keys().at( 0 );
	mPreprocessedDataPackage = *mPreprocessedDatasets.at( index ).preprocessedDataPackage; 
	mTBPAction               = mPreprocessedDatasets.at( index ).tbpActions;
	mValidationData          = validationSets.at( index ); 
}

}
