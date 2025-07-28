#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Juli 28, 2025, at 17:56
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'part0'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'PID': '',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1024, 768]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/Guess_%s_%s_%s' % (expInfo['PID'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Nemecz\\Documents\\Guess_fMRI\\part0_DE_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=1,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('hand_intro_key') is None:
        # initialise hand_intro_key
        hand_intro_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='hand_intro_key',
        )
    if deviceManager.getDevice('button_practice_key') is None:
        # initialise button_practice_key
        button_practice_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='button_practice_key',
        )
    if deviceManager.getDevice('select_video_keys') is None:
        # initialise select_video_keys
        select_video_keys = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='select_video_keys',
        )
    if deviceManager.getDevice('end_video') is None:
        # initialise end_video
        end_video = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_video',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Willkommen im MRT-Labor der Universität Regensburg!\n\nWir richten gerade den Scanner ein und starten in Kürze...\n\nVielen Dank für Ihre Teilnahme!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    # Run 'Begin Experiment' code from setup_code
    all_keys = ['space', '1','2','3','4','5']
    
    video_dict = {}
    video_dict['2'] = 'baby_animals.mp4'
    video_dict['3'] = 'deep_sea.mp4'
    video_dict['4'] = 'scotland.mp4'
    video_dict['5'] = 'brazil.mp4'
    
    
    
    # --- Initialize components for Routine "hand_intro" ---
    hand_image = visual.ImageStim(
        win=win,
        name='hand_image', 
        image='right_hand_silhouette.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.25), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    hand_intro_key = keyboard.Keyboard(deviceName='hand_intro_key')
    hand_intro_text = visual.TextStim(win=win, name='hand_intro_text',
        text='Während der Aufgaben werden Sie die Tasten mit den Fingern Ihrer rechten Hand drücken.\n\nJeder Finger wird während der Aufgabe mit einer Zahl zugeordnet.\n\nLassen Sie uns das Drücken der Tasten üben!\nWenn Sie bereit dazu sind zu üben, drücken Sie eine beliebige Taste. \n\n',
        font='Arial',
        pos=(0, 0.23), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "button_practice" ---
    instruction = visual.TextStim(win=win, name='instruction',
        text='Drücken Sie diese Tasten:',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_text = visual.TextStim(win=win, name='button_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    button_practice_key = keyboard.Keyboard(deviceName='button_practice_key')
    
    # --- Initialize components for Routine "select_video" ---
    select_video_text = visual.TextStim(win=win, name='select_video_text',
        text='Super!\n\nJetzt machen wir die strukturelle Aufnahme.\nSie können währenddessen per Tastendruck ein Video auswählen, das Sie sich anschauen möchten:\n1) Tierbabys\n2) Tiefsee\n3) Schottland\n4) Brasilien',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    select_video_keys = keyboard.Keyboard(deviceName='select_video_keys')
    
    # --- Initialize components for Routine "play_video" ---
    movie = visual.MovieStim(
        win, name='movie',
        filename=None, movieLib='ffpyplayer',
        loop=True, volume=1.0, noAudio=False,
        pos=(0, 0), size=(1024,576), units='pix',
        ori=0.0, anchor='center',opacity=None, contrast=1.0,
        depth=0
    )
    video_text = visual.TextStim(win=win, name='video_text',
        text='Das Video wird geladen...\n\nWir starten jetzt den Scanner. Bitte bleiben Sie still liegen.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    end_video = keyboard.Keyboard(deviceName='end_video')
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Der anatomische Scan ist beendet. Vielen Dank!\n\nJetzt beginnt die Aufgabe!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text, key_resp],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "hand_intro" ---
    # create an object to store info about Routine hand_intro
    hand_intro = data.Routine(
        name='hand_intro',
        components=[hand_image, hand_intro_key, hand_intro_text],
    )
    hand_intro.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for hand_intro_key
    hand_intro_key.keys = []
    hand_intro_key.rt = []
    _hand_intro_key_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for hand_intro
    hand_intro.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hand_intro.tStart = globalClock.getTime(format='float')
    hand_intro.status = STARTED
    thisExp.addData('hand_intro.started', hand_intro.tStart)
    hand_intro.maxDuration = None
    # keep track of which components have finished
    hand_introComponents = hand_intro.components
    for thisComponent in hand_intro.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "hand_intro" ---
    hand_intro.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *hand_image* updates
        
        # if hand_image is starting this frame...
        if hand_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hand_image.frameNStart = frameN  # exact frame index
            hand_image.tStart = t  # local t and not account for scr refresh
            hand_image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hand_image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hand_image.started')
            # update status
            hand_image.status = STARTED
            hand_image.setAutoDraw(True)
        
        # if hand_image is active this frame...
        if hand_image.status == STARTED:
            # update params
            pass
        
        # *hand_intro_key* updates
        waitOnFlip = False
        
        # if hand_intro_key is starting this frame...
        if hand_intro_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hand_intro_key.frameNStart = frameN  # exact frame index
            hand_intro_key.tStart = t  # local t and not account for scr refresh
            hand_intro_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hand_intro_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hand_intro_key.started')
            # update status
            hand_intro_key.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(hand_intro_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(hand_intro_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if hand_intro_key.status == STARTED and not waitOnFlip:
            theseKeys = hand_intro_key.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _hand_intro_key_allKeys.extend(theseKeys)
            if len(_hand_intro_key_allKeys):
                hand_intro_key.keys = _hand_intro_key_allKeys[-1].name  # just the last key pressed
                hand_intro_key.rt = _hand_intro_key_allKeys[-1].rt
                hand_intro_key.duration = _hand_intro_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *hand_intro_text* updates
        
        # if hand_intro_text is starting this frame...
        if hand_intro_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            hand_intro_text.frameNStart = frameN  # exact frame index
            hand_intro_text.tStart = t  # local t and not account for scr refresh
            hand_intro_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(hand_intro_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'hand_intro_text.started')
            # update status
            hand_intro_text.status = STARTED
            hand_intro_text.setAutoDraw(True)
        
        # if hand_intro_text is active this frame...
        if hand_intro_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            hand_intro.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in hand_intro.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hand_intro" ---
    for thisComponent in hand_intro.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hand_intro
    hand_intro.tStop = globalClock.getTime(format='float')
    hand_intro.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hand_intro.stopped', hand_intro.tStop)
    # check responses
    if hand_intro_key.keys in ['', [], None]:  # No response was made
        hand_intro_key.keys = None
    thisExp.addData('hand_intro_key.keys',hand_intro_key.keys)
    if hand_intro_key.keys != None:  # we had a response
        thisExp.addData('hand_intro_key.rt', hand_intro_key.rt)
        thisExp.addData('hand_intro_key.duration', hand_intro_key.duration)
    thisExp.nextEntry()
    # the Routine "hand_intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    button_practice_loop = data.TrialHandler2(
        name='button_practice_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('button_practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(button_practice_loop)  # add the loop to the experiment
    thisButton_practice_loop = button_practice_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisButton_practice_loop.rgb)
    if thisButton_practice_loop != None:
        for paramName in thisButton_practice_loop:
            globals()[paramName] = thisButton_practice_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisButton_practice_loop in button_practice_loop:
        currentLoop = button_practice_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisButton_practice_loop.rgb)
        if thisButton_practice_loop != None:
            for paramName in thisButton_practice_loop:
                globals()[paramName] = thisButton_practice_loop[paramName]
        
        # --- Prepare to start Routine "button_practice" ---
        # create an object to store info about Routine button_practice
        button_practice = data.Routine(
            name='button_practice',
            components=[instruction, button_text, button_practice_key],
        )
        button_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        button_text.setText(this_button)
        # create starting attributes for button_practice_key
        button_practice_key.keys = []
        button_practice_key.rt = []
        _button_practice_key_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'correct_button' in globals():
            correct_button = globals()['correct_button']
        # store start times for button_practice
        button_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        button_practice.tStart = globalClock.getTime(format='float')
        button_practice.status = STARTED
        thisExp.addData('button_practice.started', button_practice.tStart)
        button_practice.maxDuration = None
        # keep track of which components have finished
        button_practiceComponents = button_practice.components
        for thisComponent in button_practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "button_practice" ---
        # if trial has changed, end Routine now
        if isinstance(button_practice_loop, data.TrialHandler2) and thisButton_practice_loop.thisN != button_practice_loop.thisTrial.thisN:
            continueRoutine = False
        button_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *instruction* updates
            
            # if instruction is starting this frame...
            if instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instruction.frameNStart = frameN  # exact frame index
                instruction.tStart = t  # local t and not account for scr refresh
                instruction.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instruction, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instruction.started')
                # update status
                instruction.status = STARTED
                instruction.setAutoDraw(True)
            
            # if instruction is active this frame...
            if instruction.status == STARTED:
                # update params
                pass
            
            # *button_text* updates
            
            # if button_text is starting this frame...
            if button_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                button_text.frameNStart = frameN  # exact frame index
                button_text.tStart = t  # local t and not account for scr refresh
                button_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_text.started')
                # update status
                button_text.status = STARTED
                button_text.setAutoDraw(True)
            
            # if button_text is active this frame...
            if button_text.status == STARTED:
                # update params
                pass
            
            # *button_practice_key* updates
            waitOnFlip = False
            
            # if button_practice_key is starting this frame...
            if button_practice_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                button_practice_key.frameNStart = frameN  # exact frame index
                button_practice_key.tStart = t  # local t and not account for scr refresh
                button_practice_key.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(button_practice_key, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'button_practice_key.started')
                # update status
                button_practice_key.status = STARTED
                # allowed keys looks like a variable named `correct_button`
                if not type(correct_button) in [list, tuple, np.ndarray]:
                    if not isinstance(correct_button, str):
                        correct_button = str(correct_button)
                    elif not ',' in correct_button:
                        correct_button = (correct_button,)
                    else:
                        correct_button = eval(correct_button)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(button_practice_key.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(button_practice_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if button_practice_key.status == STARTED and not waitOnFlip:
                theseKeys = button_practice_key.getKeys(keyList=list(correct_button), ignoreKeys=["escape"], waitRelease=False)
                _button_practice_key_allKeys.extend(theseKeys)
                if len(_button_practice_key_allKeys):
                    button_practice_key.keys = _button_practice_key_allKeys[-1].name  # just the last key pressed
                    button_practice_key.rt = _button_practice_key_allKeys[-1].rt
                    button_practice_key.duration = _button_practice_key_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                button_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in button_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "button_practice" ---
        for thisComponent in button_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for button_practice
        button_practice.tStop = globalClock.getTime(format='float')
        button_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('button_practice.stopped', button_practice.tStop)
        # check responses
        if button_practice_key.keys in ['', [], None]:  # No response was made
            button_practice_key.keys = None
        button_practice_loop.addData('button_practice_key.keys',button_practice_key.keys)
        if button_practice_key.keys != None:  # we had a response
            button_practice_loop.addData('button_practice_key.rt', button_practice_key.rt)
            button_practice_loop.addData('button_practice_key.duration', button_practice_key.duration)
        # the Routine "button_practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'button_practice_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "select_video" ---
    # create an object to store info about Routine select_video
    select_video = data.Routine(
        name='select_video',
        components=[select_video_text, select_video_keys],
    )
    select_video.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for select_video_keys
    select_video_keys.keys = []
    select_video_keys.rt = []
    _select_video_keys_allKeys = []
    # allowedKeys looks like a variable, so make sure it exists locally
    if 'all_keys' in globals():
        all_keys = globals()['all_keys']
    # store start times for select_video
    select_video.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    select_video.tStart = globalClock.getTime(format='float')
    select_video.status = STARTED
    thisExp.addData('select_video.started', select_video.tStart)
    select_video.maxDuration = None
    # keep track of which components have finished
    select_videoComponents = select_video.components
    for thisComponent in select_video.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "select_video" ---
    select_video.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *select_video_text* updates
        
        # if select_video_text is starting this frame...
        if select_video_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            select_video_text.frameNStart = frameN  # exact frame index
            select_video_text.tStart = t  # local t and not account for scr refresh
            select_video_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(select_video_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'select_video_text.started')
            # update status
            select_video_text.status = STARTED
            select_video_text.setAutoDraw(True)
        
        # if select_video_text is active this frame...
        if select_video_text.status == STARTED:
            # update params
            pass
        
        # *select_video_keys* updates
        waitOnFlip = False
        
        # if select_video_keys is starting this frame...
        if select_video_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            select_video_keys.frameNStart = frameN  # exact frame index
            select_video_keys.tStart = t  # local t and not account for scr refresh
            select_video_keys.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(select_video_keys, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'select_video_keys.started')
            # update status
            select_video_keys.status = STARTED
            # allowed keys looks like a variable named `all_keys`
            if not type(all_keys) in [list, tuple, np.ndarray]:
                if not isinstance(all_keys, str):
                    all_keys = str(all_keys)
                elif not ',' in all_keys:
                    all_keys = (all_keys,)
                else:
                    all_keys = eval(all_keys)
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(select_video_keys.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(select_video_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if select_video_keys.status == STARTED and not waitOnFlip:
            theseKeys = select_video_keys.getKeys(keyList=list(all_keys), ignoreKeys=["escape"], waitRelease=False)
            _select_video_keys_allKeys.extend(theseKeys)
            if len(_select_video_keys_allKeys):
                select_video_keys.keys = _select_video_keys_allKeys[-1].name  # just the last key pressed
                select_video_keys.rt = _select_video_keys_allKeys[-1].rt
                select_video_keys.duration = _select_video_keys_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            select_video.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in select_video.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "select_video" ---
    for thisComponent in select_video.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for select_video
    select_video.tStop = globalClock.getTime(format='float')
    select_video.tStopRefresh = tThisFlipGlobal
    thisExp.addData('select_video.stopped', select_video.tStop)
    # check responses
    if select_video_keys.keys in ['', [], None]:  # No response was made
        select_video_keys.keys = None
    thisExp.addData('select_video_keys.keys',select_video_keys.keys)
    if select_video_keys.keys != None:  # we had a response
        thisExp.addData('select_video_keys.rt', select_video_keys.rt)
        thisExp.addData('select_video_keys.duration', select_video_keys.duration)
    # Run 'End Routine' code from select_video_code
    selected = select_video_keys.keys[0]
    this_video = video_dict[selected]
    print(this_video)
    thisExp.nextEntry()
    # the Routine "select_video" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "play_video" ---
    # create an object to store info about Routine play_video
    play_video = data.Routine(
        name='play_video',
        components=[movie, video_text, end_video],
    )
    play_video.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    movie.setMovie(this_video)
    # create starting attributes for end_video
    end_video.keys = []
    end_video.rt = []
    _end_video_allKeys = []
    # store start times for play_video
    play_video.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    play_video.tStart = globalClock.getTime(format='float')
    play_video.status = STARTED
    thisExp.addData('play_video.started', play_video.tStart)
    play_video.maxDuration = None
    # keep track of which components have finished
    play_videoComponents = play_video.components
    for thisComponent in play_video.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "play_video" ---
    play_video.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *movie* updates
        
        # if movie is starting this frame...
        if movie.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
            # keep track of start time/frame for later
            movie.frameNStart = frameN  # exact frame index
            movie.tStart = t  # local t and not account for scr refresh
            movie.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(movie, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'movie.started')
            # update status
            movie.status = STARTED
            movie.setAutoDraw(True)
            movie.play()
        
        # if movie is stopping this frame...
        if movie.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > movie.tStartRefresh + 600 -frameTolerance or movie.isFinished:
                # keep track of stop time/frame for later
                movie.tStop = t  # not accounting for scr refresh
                movie.tStopRefresh = tThisFlipGlobal  # on global time
                movie.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'movie.stopped')
                # update status
                movie.status = FINISHED
                movie.setAutoDraw(False)
                movie.stop()
        
        # *video_text* updates
        
        # if video_text is starting this frame...
        if video_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            video_text.frameNStart = frameN  # exact frame index
            video_text.tStart = t  # local t and not account for scr refresh
            video_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(video_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'video_text.started')
            # update status
            video_text.status = STARTED
            video_text.setAutoDraw(True)
        
        # if video_text is active this frame...
        if video_text.status == STARTED:
            # update params
            pass
        
        # if video_text is stopping this frame...
        if video_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > video_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                video_text.tStop = t  # not accounting for scr refresh
                video_text.tStopRefresh = tThisFlipGlobal  # on global time
                video_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'video_text.stopped')
                # update status
                video_text.status = FINISHED
                video_text.setAutoDraw(False)
        
        # *end_video* updates
        waitOnFlip = False
        
        # if end_video is starting this frame...
        if end_video.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_video.frameNStart = frameN  # exact frame index
            end_video.tStart = t  # local t and not account for scr refresh
            end_video.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_video, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_video.started')
            # update status
            end_video.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_video.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_video.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_video.status == STARTED and not waitOnFlip:
            theseKeys = end_video.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_video_allKeys.extend(theseKeys)
            if len(_end_video_allKeys):
                end_video.keys = _end_video_allKeys[-1].name  # just the last key pressed
                end_video.rt = _end_video_allKeys[-1].rt
                end_video.duration = _end_video_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[movie]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            play_video.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in play_video.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "play_video" ---
    for thisComponent in play_video.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for play_video
    play_video.tStop = globalClock.getTime(format='float')
    play_video.tStopRefresh = tThisFlipGlobal
    thisExp.addData('play_video.stopped', play_video.tStop)
    movie.stop()  # ensure movie has stopped at end of Routine
    # check responses
    if end_video.keys in ['', [], None]:  # No response was made
        end_video.keys = None
    thisExp.addData('end_video.keys',end_video.keys)
    if end_video.keys != None:  # we had a response
        thisExp.addData('end_video.rt', end_video.rt)
        thisExp.addData('end_video.duration', end_video.duration)
    thisExp.nextEntry()
    # the Routine "play_video" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_text, key_resp_2],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
