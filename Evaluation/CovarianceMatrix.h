#pragma once

#include <Evaluation/Export.h>
#include <DataRepresentation/Array2D.h>
#include <DataRepresentation/TabularData.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API CovarianceMatrix: public lpmldata::Array2D< double >
{

public:

	CovarianceMatrix( lpmldata::TabularData& aTabularData );

	CovarianceMatrix( lpmldata::TabularData& aFirst, lpmldata::TabularData& aSecond );

	virtual ~CovarianceMatrix();

private:

	void initialize( lpmldata::TabularData& aTabularData );

	void initialize( lpmldata::TabularData& aFirst, lpmldata::TabularData& aSecond );

};

//-----------------------------------------------------------------------------

}
