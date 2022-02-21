#pragma once

#include <Evaluation/Export.h>
#include <QSettings>
#include <QVariant>
#include <QVector>
#include <QString>
#include <QDatastream>

namespace lpmleval
{

//-----------------------------------------------------------------------------

enum class NumericType
{
	Binary = 0,
	Integer,
	Real
};

class Evaluation_API AbstractModel
{

public:
	
	AbstractModel( QSettings* aSettings );

	virtual void set( const QVector< double >& aParameters ) = 0;

	virtual QVariant evaluate( const QVector< double >& aFeatureVector ) = 0;

	virtual int inputCount() = 0;

	virtual ~AbstractModel();

	const QList< QString >& featureNames() const { return mFeatureNames; }
	QList< QString >&       featureNames() { return mFeatureNames; }

	const NumericType& numericType() const { return mNumericType; }
	NumericType&       numericType() { return mNumericType; }

	virtual void save( QDataStream& aOut ) = 0;

	virtual void load( QDataStream& aIn ) = 0;

private:

	AbstractModel();

protected:

	QSettings*        mSettings;
	QList< QString >  mFeatureNames;
	NumericType       mNumericType;
};

//-----------------------------------------------------------------------------

}

