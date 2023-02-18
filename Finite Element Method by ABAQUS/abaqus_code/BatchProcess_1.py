from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

executeOnCaeStartup()
Mdb()
Dir = 'E:/daijiabao/abaqus1000randomTi/BO_R10/'
inpDir = Dir+'inp'+'/'
rptDir = Dir+'rpt'+'/' 
odbDir = Dir+'odb'+'/' 
pre = 'Ti_BO_R10_'
preforjob='Ti_BO_R10_'
with open(odbDir+'N_1.txt',"r") as f:
    dataname=f.read()
    dataname=dataname[:-1]    
    print(dataname)
    fileName=pre+dataname
inputFileName = inpDir + fileName + '.inp'
taskName = fileName
jobname=preforjob+dataname

mdb.ModelFromInputFile(name='abaqusModel8', inputFileName=inputFileName)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['abaqusModel8'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(referenceRepresentation=OFF)
p1 = mdb.models['abaqusModel8'].parts['PART-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)

mdb.models['abaqusModel8'].Material(name='Material-1')
mdb.models['abaqusModel8'].Material(name='Material-1')
mdb.models['abaqusModel8'].materials['Material-1'].Density(table=((5.0, ), ))
mdb.models['abaqusModel8'].materials['Material-1'].Elastic(table=((46532.9, 0.34), 
    ))
mdb.models['abaqusModel8'].materials['Material-1'].Plastic(table=((919.9166667, 
    0.0), (986.4166667, 0.010345953), (1041.833333, 0.019444969), (1086.166667, 
    0.028543986), (1119.416667, 0.037643003), (1141.583333, 0.046742019), (
    1139.366667, 0.055841036), (1137.15, 0.064940052), (1131.608333, 
    0.074039069), (1124.958333, 0.083138085), (1116.091667, 0.092237102), (
    1108.333333, 0.101336118), (1075.083333, 0.110435135), (1030.75, 
    0.119534152), (980.875, 0.128633168), (908.8333333, 0.137732185), (831.25, 
    0.146831201)))


p1 = mdb.models['abaqusModel8'].parts['PART-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
mdb.models['abaqusModel8'].HomogeneousSolidSection(name='Gyroid', material='Material-1', thickness=None)
p = mdb.models['abaqusModel8'].parts['PART-1']
e = p.elements
elements = e
#elements = e.getSequenceFromMask(mask=('[#ffffffff:1000 #3 ]', ), )
region = p.Set(elements=elements, name='Gyroid')
p = mdb.models['abaqusModel8'].parts['PART-1']
p.SectionAssignment(region=region, sectionName='Gyroid', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
s = mdb.models['abaqusModel8'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(5.0, 0.0))
p = mdb.models['abaqusModel8'].Part(name='Round', dimensionality=THREE_D, 
    type=DISCRETE_RIGID_SURFACE)
p = mdb.models['abaqusModel8'].parts['Round']
p.BaseSolidExtrude(sketch=s, depth=2.0)
s.unsetPrimaryObject()
p = mdb.models['abaqusModel8'].parts['Round']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['abaqusModel8'].sketches['__profile__']
p = mdb.models['abaqusModel8'].parts['Round']
c1 = p.cells
p.RemoveCells(cellList = c1[0:1])
p = mdb.models['abaqusModel8'].parts['Round']
p.queryAttributes()
p = mdb.models['abaqusModel8'].parts['Round']
p.ReferencePoint(point=(0.0, 0.0, 0.0))
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=ON)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
p = mdb.models['abaqusModel8'].parts['Round']
p.seedPart(size=1.4, deviationFactor=0.1, minSizeFactor=0.1)
p = mdb.models['abaqusModel8'].parts['Round']
p.generateMesh()
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
    meshTechnique=OFF)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
    engineeringFeatures=ON)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=OFF)
a = mdb.models['abaqusModel8'].rootAssembly
a.regenerate()
session.viewports['Viewport: 1'].setValues(displayedObject=a)
a = mdb.models['abaqusModel8'].rootAssembly
p = mdb.models['abaqusModel8'].parts['Round']
a.Instance(name='Round-1', part=p, dependent=ON)
a = mdb.models['abaqusModel8'].rootAssembly
p = mdb.models['abaqusModel8'].parts['Round']
a.Instance(name='Round-2', part=p, dependent=ON)
a = mdb.models['abaqusModel8'].rootAssembly


if dataname=='1':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='2':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='3':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.943))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.943))
elif dataname=='4':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='5':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='6':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='7':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='8':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='9':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='10':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='11':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='12':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.943))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.943))
elif dataname=='13':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='14':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='15':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='16':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='17':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.943))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.943))
elif dataname=='18':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='19':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))
elif dataname=='20':
    a.translate(instanceList=('Round-2', ), vector=(0.0, 0.0, -1.9485))
    a = mdb.models['abaqusModel8'].rootAssembly
    a.translate(instanceList=('Round-1', ), vector=(0.0, 0.0, 5.9485))



a = mdb.models['abaqusModel8'].rootAssembly
a.translate(instanceList=('PART-1-1', ), vector=(-3, -3, 0.0))
a = mdb.models['abaqusModel8'].rootAssembly

session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
mdb.models['abaqusModel8'].ExplicitDynamicsStep(name='Step-1', 
    previous='Initial', timePeriod=7.333, improvedDtMethod=ON)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
mdb.models['abaqusModel8'].steps['Step-1'].setValues(massScaling=PREVIOUS_STEP, improvedDtMethod=ON)
mdb.models['abaqusModel8'].FieldOutputRequest(name='F-Output-2', 
    createStepName='Step-1', variables=('U', 'V', 'A', 'RF', 'RT'))
mdb.models['abaqusModel8'].fieldOutputRequests['F-Output-2'].setValues(
numIntervals=22)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(interactions=ON, 
    constraints=ON, connectors=ON, engineeringFeatures=ON, 
    adaptiveMeshConstraints=OFF)
mdb.models['abaqusModel8'].ContactProperty('IntProp-1')
mdb.models['abaqusModel8'].interactionProperties['IntProp-1'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    0.125, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)

mdb.models['abaqusModel8'].ContactExp(name='Int-1', createStepName='Step-1')
mdb.models['abaqusModel8'].interactions['Int-1'].includedPairs.setValuesInStep(
    stepName='Step-1', useAllstar=ON)
mdb.models['abaqusModel8'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
    stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))

session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
    predefinedFields=ON, interactions=OFF, constraints=OFF, 
    engineeringFeatures=OFF)
session.viewports['Viewport: 1'].view.setValues(nearPlane=26.3023, 
    farPlane=45.0846, width=21.8334, height=12.6099, cameraPosition=(-31.9957, 
    -4.25204, -12.2387), cameraUpVector=(0.334762, -0.884768, 0.324222), 
    cameraTarget=(-0.693402, -0.101584, 2.66634))
session.viewports['Viewport: 1'].view.setValues(nearPlane=25.7079, 
    farPlane=45.6825, width=21.34, height=12.325, cameraPosition=(-23.941, 
    -6.20642, -22.7382), cameraUpVector=(0.110802, -0.822844, 0.55736), 
    cameraTarget=(-0.51826, -0.14408, 2.43804))
a = mdb.models['abaqusModel8'].rootAssembly
r1 = a.instances['Round-2'].referencePoints
refPoints1=(r1[3], )
region = a.Set(referencePoints=refPoints1, name='Set-2')
mdb.models['abaqusModel8'].DisplacementBC(name='BC-1', createStepName='Step-1', 
    region=region, u1=0.0, u2=0.0, u3=0.0, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
mdb.models['abaqusModel8'].TabularAmplitude(name='Amp-1', timeSpan=STEP, 
    smooth=SOLVER_DEFAULT, data=((0.0, 0.0), (7.333, 1.0)))
a = mdb.models['abaqusModel8'].rootAssembly
r1 = a.instances['Round-1'].referencePoints
refPoints1=(r1[3], )
region = a.Set(referencePoints=refPoints1, name='Set-3')
mdb.models['abaqusModel8'].DisplacementBC(name='BC-2', createStepName='Step-1', 
    region=region, u1=0.0, u2=0.0, u3=-0.36667, ur1=0.0, ur2=0.0, ur3=0.0, 
    amplitude='Amp-1', fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)
mdb.models['abaqusModel8'].fieldOutputRequests['F-Output-1'].setValues(
numIntervals=1)
session.viewports['Viewport: 1'].view.setValues(nearPlane=27.354, 
    farPlane=44.0263, width=22.7064, height=13.1142, cameraPosition=(-35.095, 
    5.90566, 0.307251), cameraUpVector=(0.18363, -0.975823, -0.118535), 
    cameraTarget=(-0.761262, 0.119269, 2.94))

mdb.Job(name=jobname, model='abaqusModel8', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
    nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
    contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
    resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=56, 
    activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=56)
mdb.jobs[jobname].submit(consistencyChecking=OFF)
mdb.jobs[jobname].waitForCompletion()
del mdb.models['abaqusModel8']
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
Mdb()
odbFileName =odbDir+ jobname + '.odb'
o1 = session.openOdb(name=odbFileName)
session.animationController.setValues(animationType=SCALE_FACTOR, viewports=('Viewport: 1', ))
session.animationController.play(duration=UNLIMITED)
session.animationController.animationOptions.setValues(frameRate=28)
session.animationController.stop()
odb = session.odbs[odbFileName]
session.viewports['Viewport: 1'].setValues(displayedObject=odb)
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF3'), )), ('U', NODAL, ((COMPONENT, 'U3'), )), ), 
    nodeSets=('REFERENCE_POINT_ROUND-1      133', ))
xy1 = session.xyDataObjects['U:U3 PI: ROUND-1 N: 133']
xy2 = session.xyDataObjects['RF:RF3 PI: ROUND-1 N: 133']
xy3 = combine(-xy1, -xy2)
tmpName = xy3.name
XYPlotName='name'
xyp = session.XYPlot(XYPlotName)
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
c1 = session.Curve(xyData=xy3)
chart.setValues(curvesToPlot=(c1, ), )
xyp = session.xyPlots[XYPlotName]
chartName = xyp.charts.keys()[0]
chart = xyp.charts[chartName]
x0 = chart.curves[tmpName]
session.writeXYReport(fileName=rptDir + taskName +'.rpt', xyData=(x0, ))
session.odbs[odbFileName].close()
executeOnCaeStartup()