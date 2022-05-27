## REU Computing for Structure: 2022

The website for the project is at https://www.cs.miami.edu/reu-cfs.

- Funding for summers 2017&ndash;2019 by NSF grant CNS-1659144. 
- The renewal grant CNS-1949972 was postponed for Covid in 2020 and resumes for years 2021&ndash;2023.

Our mentors this year are, 

- Prof. Orlando Acevado 
- Prof. Vance Lemmon
- Prof. Zhang Wang 
- Prof. Daniel Messinger
- Laura Vitale 
- Prof. Vanessa Aguiar-Pulido 

### SSH Public Key

Your Cane ID is your name you received at the Cane ID website. To manage or reset your password, visit https://caneidhelp.miami.edu/caneid/.

We refer to the password you used to create this account, as the Cane ID password. You have recieved other passwords for other machines,
such as the lab machines and for Pegasus. 

To help alleiviate the burden of passwords and other troublesome identifiers, as well as to enhance security, public key authentication is preferred
for ssh. It requires you create a public/private key pair using the command line program `ssh-keygen`. You can use it without parameters.
It will prompt for the name of the file to create. I suggesst `id_rsa_triton`, although this is not critical. 

The program will create two files: `id_rsa_triton` and `id_rsa_triton.pub`. The first contains the private key and <u>must</u> be kept secret.
The .pub file is the public key and you can share that freely. 

<menu>
The public key system will be a conversation between the public key holder and 
the private key holder during which the public key holder becomes convinced that the counter-party  knows the contents of the private key, 
and hence (under favorable assumptions) is authentic. However, this converstation reveals nothing about the private key to any party, not
  to an eavesdropper nor to the holder of the public key.
</menu>


1. Logon onto Triton. 
