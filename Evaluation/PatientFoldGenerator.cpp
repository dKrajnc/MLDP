#include "Evaluation/PatientFoldGenerator.h"
#include <random>
#include <qdebug.h>

namespace dkeval
{

//-----------------------------------------------------------------------------

PatientFoldGenerator::PatientFoldGenerator( lpmldata::DataPackage aDataPackage, int aMinSubsampleCount )
:
	mDataPackage( aDataPackage ),
	mMinSubsampleCount( aMinSubsampleCount ),
	mPatientNames()
{
	auto keys = mDataPackage.sampleKeys();

	for ( auto key : keys )
	{
		auto splittedKey = key.split( "/Scan-" );
		mPatientNames.push_back( splittedKey.at( 0 ) );
	}
}

//-----------------------------------------------------------------------------

void PatientFoldGenerator::generate( int aFoldCount )
{
	std::random_device rd;
	std::mt19937 g( rd() );

	int maxAttemptCount = 0;

	auto minorityKeys   = mDataPackage.getMinorityKeys();
	auto majorityKeys   = mDataPackage.getMajorityKeys();	
	int subgroupCount   = static_cast< int >( mMinSubsampleCount / 2 ); //get number of keys per subgroup which constructs validation set
	
	for ( auto key : minorityKeys )
	{
		auto index = minorityKeys.indexOf( key );
		QString splittedKey = key.split( "/Scan-" ).at( 0 );
		minorityKeys.replace( index, splittedKey );

	}
	for ( auto key : majorityKeys )
	{
		auto index = majorityKeys.indexOf( key );
		QString splittedKey = key.split( "/Scan-" ).at( 0 );
		majorityKeys.replace( index, splittedKey );

	}

	auto validationMinoritiesCount = static_cast< int >( ( minorityKeys.size() * 20 ) / 100 ); //ensure 20% split for validation with respect to minority subclass
	if ( validationMinoritiesCount < 1 ) validationMinoritiesCount = 1;  //ensure atlesat 1 minority sample in validation data
	subgroupCount = validationMinoritiesCount;
	mMinSubsampleCount = subgroupCount * 2;
	qDebug() << "Validation set size:" << mMinSubsampleCount;

	while ( mValidationPatientHistory.size() < aFoldCount )
	{						
		std::shuffle( minorityKeys.begin(), minorityKeys.end(), g );
		std::shuffle( majorityKeys.begin(), majorityKeys.end(), g );

		QList< QString > validationPatients;
		QList< QString > trainingPatients;

		validationPatients = minorityKeys.mid( 0, subgroupCount );
		trainingPatients   = minorityKeys.mid( subgroupCount, minorityKeys.size() );
		validationPatients.append( majorityKeys.mid( 0, subgroupCount ) );
		trainingPatients.append( majorityKeys.mid( subgroupCount, majorityKeys.size() ) );


		if ( isValidValidationSet( validationPatients ) )
		{
			if ( !mValidationPatientHistory.contains( validationPatients ) )
			{
				if ( isValidTrainingSet( trainingPatients ) )
				{
					mValidationPatientHistory.push_back( validationPatients );
				}
				else
				{
					qDebug() << "Found a new valdiation set but training set is invalid - skip";
					++maxAttemptCount;
				}
			}
			else
			{
				qDebug() << "Found a duplicate validation set - skip";
				++maxAttemptCount;
			}

			if ( maxAttemptCount > aFoldCount )
			{
				qDebug() << "Sorry dude, too many trials - dataset is problematic? That is what you get:" << mValidationPatientHistory.size();
				break;
			}
		}
	}
}

//-----------------------------------------------------------------------------

Pair PatientFoldGenerator::fold( int aFoldIndex )
{
	lpmldata::TabularData validationFDB;
	lpmldata::TabularData validationLDB;
	lpmldata::TabularData trainingFDB;
	lpmldata::TabularData trainingLDB;

	if ( aFoldIndex < mValidationPatientHistory.size() )
	{
		auto validationPatients = mValidationPatientHistory.at( aFoldIndex );
		auto keys = mDataPackage.sampleKeys();

		QList< QString > validationKeys;
		for ( auto key : keys )
		{
			auto patientKey = key.split( "/Scan-" ).at( 0 );

			if ( validationPatients.contains( patientKey ) )
			{
				validationKeys.push_back( key );
			}
		}

		validationFDB.setHeader( mDataPackage.featureDatabase().headerNames() );
		validationLDB.setHeader( mDataPackage.labelDatabase().headerNames() );
		
		trainingFDB.setHeader( mDataPackage.featureDatabase().headerNames() );
		trainingLDB.setHeader( mDataPackage.labelDatabase().headerNames() );

		for ( auto key : keys )
		{
			auto featureVector = mDataPackage.featureDatabase().value( key );
			auto labelVector   = mDataPackage.labelDatabase().value( key );

			if ( validationKeys.contains( key ) )
			{
				validationFDB.insert( key, featureVector );
				validationLDB.insert( key, labelVector );
			}
			else
			{
				trainingFDB.insert( key, featureVector );
				trainingLDB.insert( key, labelVector );
			}
		}
	}
	
	std::shared_ptr< lpmldata::DataPackage > trainingDP = std::make_shared< lpmldata::DataPackage >( trainingFDB, trainingLDB, mDataPackage.labelName() );
	trainingDP->setActiveLabelIndex( 0 );

	std::shared_ptr< lpmldata::DataPackage > validationDP = std::make_shared< lpmldata::DataPackage >( validationFDB, validationLDB, mDataPackage.labelName() );
	validationDP->setActiveLabelIndex( 0 );

	return Pair( trainingDP, validationDP );
}

//-----------------------------------------------------------------------------

bool PatientFoldGenerator::isValidTrainingSet( const QList<QString>& aTrainingKeys )
{
	QMap< QString, int > trainingSubgroupCounts;
	auto keys = mDataPackage.sampleKeys();

	for ( auto key : keys )
	{
		auto patientKey = key.split( "/Scan-" ).at( 0 );
		if ( aTrainingKeys.contains( patientKey ) )
		{
			QString labelOfKey = mDataPackage.labelDatabase().value( key ).at( mDataPackage.labelIndex() ).toString();
			++trainingSubgroupCounts[ labelOfKey ];
		}	
	}

	bool isValidDataset = true;
	for ( auto label : trainingSubgroupCounts.keys() )
	{
		if ( trainingSubgroupCounts.value( label ) < mMinSubsampleCount ) isValidDataset = false;
	}

	return isValidDataset;
}

//-----------------------------------------------------------------------------

bool PatientFoldGenerator::isValidValidationSet( const QList< QString >& aValidationKeys )
{
	bool isValidDataset          = true;
	int firstGroupMemberCounter  = 0;
	int secondGroupMemberCounter = 0;

	QList< QString > firstGroupKeys;
	QList< QString > secondGroupKeys;
	

	auto LDB         = mDataPackage.labelDatabase();
	auto labelgroups = mDataPackage.labelOutcomes();

	for ( int i = 0; i < LDB.keys().size(); ++i )
	{
		auto key         = LDB.keys().at( i );
		auto splittedKey = key.split( "/Scan-" ).at( 0 );
		auto label       = LDB.valueAt( key, 0 );

		if ( mPatientNames.contains( splittedKey ) )
		{
			if ( label == labelgroups.at( 0 ) )
			{
				firstGroupKeys << splittedKey;
			}
			else if ( label == labelgroups.at( 1 ) )
			{
				secondGroupKeys << splittedKey;
			}
		}
	}

	for ( auto& key : aValidationKeys )
	{
		if ( firstGroupKeys.contains( key ) )
		{
			firstGroupMemberCounter++;
		}
		else if ( secondGroupKeys.contains( key ) )
		{
			secondGroupMemberCounter++;
		}
	}

	if ( firstGroupMemberCounter != 0 && secondGroupMemberCounter != 0 )
	{
		return true;
	}
	else return false;
}

//-----------------------------------------------------------------------------

}