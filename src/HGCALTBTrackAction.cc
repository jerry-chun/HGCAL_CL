//**************************************************
// \file HGCALTBTrackAction.cc
// \brief: Implementation of HGCALTBTrackAction class
// \author: Lorenzo Pezzotti (CERN EP-SFT-sim)
// \start date: 15 January 2024
//**************************************************

#include "HGCALTBTrackAction.hh"
#include "TrackPrimaryMap.hh"
#include "G4DynamicParticle.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4VProcess.hh"

#ifdef USE_CELERITAS
#include "Celeritas.hh"
#endif

#include <string>
#include <iostream>

HGCALTBTrackAction::HGCALTBTrackAction(HGCALTBEventAction* EvtAction)
  : G4UserTrackingAction(), fEventAction(EvtAction)
{}

HGCALTBTrackAction::~HGCALTBTrackAction() {}

// Called at the start of every new track
void HGCALTBTrackAction::PreUserTrackingAction(const G4Track* aTrack) {
  G4int trackID = aTrack->GetTrackID();
  G4int parentID = aTrack->GetParentID();

  if (parentID == 0) {
    // Assign a new shower index
    gTrackToPrimaryMap[trackID] = currentShowerIndex;
    G4cout << "[TrackAction] Primary trackID " << trackID
           << " ? showerID " << currentShowerIndex << G4endl;
    currentShowerIndex++;
  } else {
    auto it = gTrackToPrimaryMap.find(parentID);
    if (it != gTrackToPrimaryMap.end()) {
      gTrackToPrimaryMap[trackID] = it->second;
    } else {
      G4cerr << "[TrackAction] Warning: parentID " << parentID
             << " not in gTrackToPrimaryMap for trackID " << trackID << G4endl;
    }
  }

 // G4cout << "[TrackAction] trackID " << trackID
 //        << " assigned showerID " << gTrackToPrimaryMap[trackID] << G4endl;
}

void HGCALTBTrackAction::PostUserTrackingAction(const G4Track* aTrack)
{
  if (aTrack->GetParentID() == 0) {
    const G4Step* step = aTrack->GetStep();
    if (!step) return;

    const G4StepPoint* poststep = step->GetPostStepPoint();
    if (!poststep) return;

    const G4VProcess* process = poststep->GetProcessDefinedStep();
    if (!process) return;

    // Subtype 121: Hadronic inelastic interaction
    if (process->GetProcessSubType() == 121) {
      G4String volName = step->GetPreStepPoint()->GetTouchableHandle()->GetVolume()->GetName();
      G4String CEEsubname = "HGCalEE";
      G4int InteractionLayer = (volName.find(CEEsubname) != std::string::npos) ? 1 : 0;
      fEventAction->SetIntLayer(InteractionLayer);
    }
  }
}
